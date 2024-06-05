#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, avgpool_forward)
{
    /* Initialize params */
    ns_AveragePooling::Params kernel_params;
    memset(&kernel_params, 0, sizeof(ns_AveragePooling::Params));
    kernel_params.includePadding = 0;
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
    const unsigned inZ     = 2;
    const unsigned inW     = 4;
    const unsigned inH     = 4;
    const unsigned batch   = 1;

    float inputData [batch][inH][inW][inZ] =
            {{
                     {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
                     {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
                     {{7, 6}, {5, 4}, {3, 2}, {1, 0}},
                     {{7, 6}, {5, 4}, {3, 2}, {1, 0}}
             }};

    float outputRef [batch][3][3][2] =
            {{
                     {{1, 2}, {3, 4}, {5, 6}},
                     {{3.5, 3.5}, {3.5, 3.5}, {3.5, 3.5}},
                     {{6, 5}, {4, 3}, {2, 1}}
             }};

    unsigned outW =
        (inW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) / kernel_params.stride_w + 1;
    unsigned outH =
        (inH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1) / kernel_params.stride_h + 1;

    // Tensor size [NHWC]
    unsigned inTensorSize[SYN_MAX_TENSOR_DIM]  = {inZ, inW, inH, 1};
    unsigned outTensorSize[SYN_MAX_TENSOR_DIM] = {inZ, outW, outH, 1};

    unsigned outputBufferSize = outTensorSize[0] * outTensorSize[1] * outTensorSize[2] * outTensorSize[3];

    auto firstTensorIndex = createPersistTensor(TensorUsage::INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                reinterpret_cast<float*>(inputData),
                                                inTensorSize,
                                                4,
                                                asSynType<float>());
    auto out = createPersistTensor(
        TensorUsage::OUTPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, outTensorSize, 4, asSynType<float>());

    char nodeGuid[] = "avg_pool_2d_fwd_f32";
    addNodeToGraph(nodeGuid, {firstTensorIndex}, {out}, (void*)&kernel_params, sizeof(ns_AveragePooling::Params));
    compileAndRun();

    // validate results
    float* outputBuffer = castHostOutBuffer<float>(out);
    float* pOutputRef   = (float*)outputRef;
    for (unsigned i = 0; i < outputBufferSize; ++i)
    {
        ASSERT_EQ(outputBuffer[i], pOutputRef[i]) << "Mismatch for output-index " << 0 << " at index " << i
                                                  << " Expected:" << pOutputRef[i] << " Result: " << outputBuffer[i];
    }
}

TEST_F_GC(SynTrainingTestInfra, avgpool_forwardn_bf16)
{
    /* Initialize params */
    ns_AveragePooling::Params kernel_params;
    memset(&kernel_params, 0, sizeof(ns_AveragePooling::Params));
    kernel_params.includePadding = 0;
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
    const unsigned inZ     = 2;
    const unsigned inW     = 4;
    const unsigned inH     = 4;
    const unsigned batch   = 1;

    float inputData [batch][inH][inW][inZ] =
        {{
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
            {{7, 6}, {5, 4}, {3, 2}, {1, 0}},
            {{7, 6}, {5, 4}, {3, 2}, {1, 0}}
        }};

    bfloat16 outputRef [batch][3][3][2] =
        {{
            {{1, 2}, {3, 4}, {5, 6}},
            {{3.5, 3.5}, {3.5, 3.5}, {3.5, 3.5}},
            {{6, 5}, {4, 3}, {2, 1}}
        }};
    UNUSED(outputRef);

    unsigned outW = (inW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1)/ kernel_params.stride_w + 1;
    unsigned outH = (inH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1)/ kernel_params.stride_h + 1;

    // Tensor size [NHWC]
    unsigned inTensorSize[SYN_MAX_TENSOR_DIM]   = { inZ, inW, inH, 1 };
    unsigned outTensorSize[SYN_MAX_TENSOR_DIM]  = { inZ, outW, outH, 1 };

    auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, reinterpret_cast<float*>(inputData), inTensorSize, 4, asSynType<bfloat16>());
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outTensorSize, 4, asSynType<bfloat16>());
    char nodeGuid[] = "avg_pool_2d_fwd_bf16";
    addNodeToGraph( nodeGuid, {in}, {out}, &kernel_params, sizeof(ns_AveragePooling::Params));

    compileAndRun();

    // validate results
    auto* pOutputBuffer = castHostOutBuffer<bfloat16>(out);
    auto* pOutputRef    = (bfloat16*)outputRef;
    for (uint64_t i = 0; i < getTensorElementCount(out); ++i)
    {
        ASSERT_EQ(pOutputBuffer->value(), pOutputRef->value())
            << "Mismatch for output-index " << 0 << " at index " << i << " Expected:" << pOutputRef->value()
            << " Result: " << pOutputBuffer->value();
        pOutputBuffer++;
        pOutputRef++;
    }
}

TEST_F_GC(SynTrainingTestInfra, avgpool_backward)
{
    /* Initialize params */
    ns_AveragePooling::Params kernel_params;
    memset(&kernel_params, 0, sizeof(ns_AveragePooling::Params));
    kernel_params.includePadding = 0;
    kernel_params.pad_w_begin    = 0;
    kernel_params.pad_h_end      = 0;
    kernel_params.pad_w_end      = 0;
    kernel_params.pad_h_begin    = 0;
    kernel_params.kernel_w       = 7;
    kernel_params.kernel_h       = 7;
    kernel_params.stride_w       = 1;
    kernel_params.stride_h       = 1;
    kernel_params.dilation_w     = 1;
    kernel_params.dilation_h     = 1;

    /* const params */
    const unsigned inZ     = 1;
    const unsigned inW     = 1;
    const unsigned inH     = 1;
    const unsigned batch   = 1;

    float inputData [batch][inH][inW][inZ] =
            {{
                    {{49}}
            }};

    float outputRef [batch][7][7][1] =
            {{
                     {{1}, {1}, {1}, {1}, {1}, {1}, {1}},
                     {{1}, {1}, {1}, {1}, {1}, {1}, {1}},
                     {{1}, {1}, {1}, {1}, {1}, {1}, {1}},
                     {{1}, {1}, {1}, {1}, {1}, {1}, {1}},
                     {{1}, {1}, {1}, {1}, {1}, {1}, {1}},
                     {{1}, {1}, {1}, {1}, {1}, {1}, {1}},
                     {{1}, {1}, {1}, {1}, {1}, {1}, {1}},
             }};
    //*/
    unsigned outW = ((inW - 1) * kernel_params.stride_w) - (kernel_params.pad_w_begin + kernel_params.pad_w_end) + (kernel_params.dilation_w * (kernel_params.kernel_w-1)) + 1;
    unsigned outH = ((inH - 1) * kernel_params.stride_h) - (kernel_params.pad_h_begin + kernel_params.pad_h_end) + (kernel_params.dilation_h * (kernel_params.kernel_h-1)) + 1;

    // Tensor size [NHWC]
    unsigned inTensorSize[SYN_MAX_TENSOR_DIM]   = { inZ, inW, inH, 1 };
    unsigned outTensorSize[SYN_MAX_TENSOR_DIM]  = { inZ, outW, outH, 1 };

    createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, reinterpret_cast<float*>(inputData), inTensorSize, 4, asSynType<float>());
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outTensorSize, 4, asSynType<float>());

    char nodeGuid[] = "avg_pool_2d_bwd_f32";
    addNodeToGraph(nodeGuid, &kernel_params, sizeof(ns_AveragePooling::Params));
    compileAndRun();

    // validate results
    float* pOutputBuffer = castHostOutBuffer<float>(out);
    float* pOutputRef    = (float*)outputRef;
    for (uint64_t i = 0; i < getTensorElementCount(out); ++i)
    {
        ASSERT_EQ( *pOutputBuffer, *pOutputRef ) << "Mismatch at index " << i
                                                 << " Expected:"         << *pOutputRef
                                                 << " Result: "          << *pOutputBuffer << std::endl;
        pOutputBuffer++;
        pOutputRef++;
    }
}

TEST_F_GC(SynTrainingTestInfra, avgpool_forward_no_aux_provided)
{
    /* Initialize params */
    ns_AveragePooling::Params kernel_params;
    memset(&kernel_params, 0, sizeof(ns_AveragePooling::Params));
    kernel_params.includePadding = 0;
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
    const unsigned inZ     = 2;
    const unsigned inW     = 4;
    const unsigned inH     = 4;
    const unsigned batch   = 1;

    float inputData [batch][inH][inW][inZ] =
            {{
                     {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
                     {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
                     {{7, 6}, {5, 4}, {3, 2}, {1, 0}},
                     {{7, 6}, {5, 4}, {3, 2}, {1, 0}}
             }};

    float outputRef [batch][3][3][2] =
            {{
                     {{1, 2}, {3, 4}, {5, 6}},
                     {{3.5, 3.5}, {3.5, 3.5}, {3.5, 3.5}},
                     {{6, 5}, {4, 3}, {2, 1}}
             }};
    UNUSED(outputRef);

    unsigned outW = (inW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1)/ kernel_params.stride_w + 1;
    unsigned outH = (inH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1)/ kernel_params.stride_h + 1;

    // Tensor size [NHWC]
    unsigned inTensorSize[SYN_MAX_TENSOR_DIM]  = {inZ, inW, inH, 1};
    unsigned outTensorSize[SYN_MAX_TENSOR_DIM] = {inZ, outW, outH, 1};

    createPersistTensor(INPUT_TENSOR,
                        MEM_INIT_FROM_INITIALIZER,
                        reinterpret_cast<float*>(inputData),
                        inTensorSize,
                        4,
                        asSynType<float>());

    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outTensorSize, 4, asSynType<float>());

    char nodeGuid[] = "avg_pool_2d_fwd_f32";
    addNodeToGraph(nodeGuid, &kernel_params, sizeof(ns_AveragePooling::Params));
    compileAndRun();

    // validate results
    float* pOutputBuffer = castHostOutBuffer<float>(out); // max values
    float* pOutputRef    = (float*)outputRef;
    for (uint64_t i = 0; i < getTensorElementCount(out); ++i)
    {
        ASSERT_EQ(*pOutputBuffer, *pOutputRef) << "Mismatch for output-index " << 0 << " at index " << i
                                               << " Expected:" << *pOutputRef << " Result: " << *pOutputBuffer;
        pOutputBuffer++;
        pOutputRef++;
    }
}
