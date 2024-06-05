#include "gc_dynamic_shapes_infra.h"
#include "node_factory.h"
#include "tpc_batch_norm_test.h"
#include "vtune_stat.h"

class SynGaudiDynamicShapesMilestone2 : public SynGaudiDynamicShapesTestsInfra,
                                        public testing::WithParamInterface<bool>
{
public:

    static const size_t C = 16;
    static const size_t MAX_W = 128;
    static const size_t MIN_W = 64;
    static const size_t MAX_H = 128;
    static const size_t MIN_H = 64;
    static const size_t BATCH = 16;
    static const size_t TENSOR_DIMS = 4;
    static const size_t COPY_TESNOR_NR = 2;
    static const size_t SPLIT_DIM = 3;

protected:
    virtual void SetUpTest() override
    {
        if (GetParam())
        {
            GCFG_ENABLE_STAGED_SUBMISSION.setValue(true);
        }
        SynGaudiDynamicShapesTestsInfra::SetUpTest();
    }
};


INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicShapesMilestone2, ::testing::Values(true, false));

// This test is testing dynamic shapes milestone 2 logic. Check out [SW-20642] in the jira to find out the full test
// layout. But simply its doing :
//
//     Bias  Bias           Data
//       \    /               |
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                        Max Pool 2D           Bias
//                           Relu             Broadcast
//                            |                   |
//                           Add                  |
//                          Split <----------------
//                          |   |
//                     Memcopy  Memcopy
//                          \   /
//                          Concat

TEST_P_GC(SynGaudiDynamicShapesMilestone2, ms2_test, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    if (GetParam())
    {
        HB_ASSERT(GCFG_ENABLE_STAGED_SUBMISSION.value() == true, "ENABLE_STAGED_SUBMISSION should be true for this test variant");
    }
    size_t inActualW = 100;
    size_t inActualH = 92;

    synConvolutionParams convParams;
    convParams.dH   = 1;
    convParams.dW   = 1;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.dilH = 1;
    convParams.dilW = 1;
    convParams.padT = 0;
    convParams.padB = 0;
    convParams.padL = 0;
    convParams.padR = 0;

    // Create the static part of the graph that is inserted as weights to the conv.
    unsigned convWeightsSize[] = {C, C, convParams.kW, convParams.kH};

    unsigned weightsPart1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                convWeightsSize, TENSOR_DIMS, syn_type_float);

    unsigned copiedWeightsPart1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                               convWeightsSize, TENSOR_DIMS, syn_type_float);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {weightsPart1}, {copiedWeightsPart1});

    unsigned weightsPart2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                convWeightsSize, TENSOR_DIMS, syn_type_float);

    unsigned copiedWeightsPart2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                               convWeightsSize, TENSOR_DIMS, syn_type_float);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {weightsPart2}, {copiedWeightsPart2});

    unsigned weightsFinal = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                         convWeightsSize, TENSOR_DIMS, syn_type_float);

    addNodeToGraph("add_fwd_f32", {copiedWeightsPart1, copiedWeightsPart2}, {weightsFinal});

    // Create main branch
    unsigned inMaxSizes[] = {C, MAX_W, MAX_H, BATCH};
    unsigned inMinSizes[] = {C, MIN_W, MIN_H, BATCH};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, inMinSizes);

    const unsigned convOutMaxW = convOutputDimSize(MAX_W, convParams.kW, convParams.dW,
                                                   convParams.padL + convParams.padR, convParams.dilW);
    const unsigned convOutMinW = convOutputDimSize(MIN_W, convParams.kW, convParams.dW,
                                                   convParams.padL + convParams.padR, convParams.dilW);
    const unsigned convOutActualW = convOutputDimSize(inActualW, convParams.kH, convParams.dH,
                                                      convParams.padT + convParams.padB, convParams.dilH);
    const unsigned convOutMaxH = convOutputDimSize(MAX_H, convParams.kH, convParams.dH,
                                                   convParams.padT + convParams.padB, convParams.dilH);
    const unsigned convOutMinH = convOutputDimSize(MIN_H, convParams.kH, convParams.dH,
                                                   convParams.padT + convParams.padB, convParams.dilH);
    const unsigned convOutActualH = convOutputDimSize(inActualH, convParams.kH, convParams.dH,
                                                      convParams.padT + convParams.padB, convParams.dilH);

    unsigned convOutMaxSizes[] = {C, convOutMaxW, convOutMaxH, BATCH};
    unsigned convOutMinSizes[] = {C, convOutMinW, convOutMinH, BATCH};
    unsigned convOutActualSizes[] = {C, convOutActualW, convOutActualH, BATCH};

    unsigned bnInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                            convOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, convOutMinSizes);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {inTensor, weightsFinal},
                   {bnInTensor}, (void*)&convParams, sizeof(synConvolutionParams));

    unsigned bnTensorSizes[] = {C};

    unsigned bnBetaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                bnTensorSizes, 1, syn_type_float, nullptr, "Beta");
    unsigned bnGammaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                 bnTensorSizes, 1, syn_type_float, nullptr, "Gamma");
    unsigned bnInMeanTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                  bnTensorSizes, 1, syn_type_float, nullptr, "In Mean");
    unsigned bnInVarTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                 bnTensorSizes, 1, syn_type_float, nullptr, "In Var");

    unsigned maxpoolInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        convOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, convOutMinSizes);

    unsigned bnOutMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                            bnTensorSizes, 1, syn_type_float);
    unsigned bnOutStdTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                           bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                               bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunVarTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                              bnTensorSizes, 1, syn_type_float);

    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;

    addNodeToGraph("batch_norm_fwd_f32", {bnInTensor, bnBetaTensor, bnGammaTensor, bnInMeanTensor, bnInVarTensor},
                   {maxpoolInTensor, bnOutMeanTensor, bnOutStdTensor, bnOutRunMeanTensor, bnOutRunVarTensor},
                   &bnParams, sizeof(ns_BatchNormKernel::Params));

    ns_SpatialReduction::Params kernel_params;
    kernel_params.pad_w_begin = 0;
    kernel_params.pad_h_end   = 0;
    kernel_params.pad_w_end   = 0;
    kernel_params.pad_h_begin = 0;
    kernel_params.kernel_w    = 2;
    kernel_params.kernel_h    = 2;
    kernel_params.stride_w    = 2;
    kernel_params.stride_h    = 2;
    kernel_params.dilation_w  = 1;
    kernel_params.dilation_h  = 1;
    kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    unsigned poolOutMaxW = (convOutMaxW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                                kernel_params.stride_w + 1;
    unsigned poolOutMinW = (convOutMinW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                                kernel_params.stride_w + 1;
    unsigned poolOutActualW = (convOutActualW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                                kernel_params.stride_w + 1;

    unsigned poolOutMaxH = (convOutMaxH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1) /
                                kernel_params.stride_h + 1;
    unsigned poolOutMinH = (convOutMinH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1) /
                           kernel_params.stride_h + 1;
    unsigned poolOutActualH = (convOutActualH + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                              kernel_params.stride_w + 1;

    unsigned maxpoolOutMaxSizes[] = {C, poolOutMaxW, poolOutMaxH, BATCH};
    unsigned maxpoolOutMinSizes[] = {C, poolOutMinW, poolOutMinH, BATCH};
    unsigned maxpoolOutActualSizes[] = {C, poolOutActualW, poolOutActualH, BATCH};

    unsigned maxpoolRetainTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                maxpoolOutMaxSizes, TENSOR_DIMS, syn_type_uint8, nullptr, maxpoolOutMinSizes);

    unsigned addInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        maxpoolOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, maxpoolOutMinSizes);

    addNodeToGraph("maxpool_2d_fwd_f32", {maxpoolInTensor},
                   {maxpoolRetainTensor, addInTensor}, &kernel_params, sizeof(ns_SpatialReduction::Params));

    unsigned biasSizes[] = {C, 1, 1, 1};
    unsigned biasTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr,
                                              biasSizes, TENSOR_DIMS, syn_type_float,nullptr, "Bias");

    unsigned biasShape = createShapeTensor(INPUT_TENSOR, maxpoolOutMaxSizes, maxpoolOutMinSizes, TENSOR_DIMS,
                                           syn_type_float, "Broadcast Size", 0);

    unsigned broadcastOutTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                      maxpoolOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, maxpoolOutMinSizes);

    addNodeToGraph(NodeFactory::broadcastNodeTypeName, {biasTensor, biasShape}, {broadcastOutTensor});

    unsigned reluInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                         maxpoolOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, maxpoolOutMinSizes);

    addNodeToGraph("add_fwd_f32", {addInTensor, broadcastOutTensor}, {reluInTensor});


    unsigned splitInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                          maxpoolOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, maxpoolOutMinSizes);

    addNodeToGraph("relu_fwd_f32", {reluInTensor}, {splitInTensor});

    unsigned copiedReluMaxSizes[] = {C, maxpoolOutMaxSizes[1], maxpoolOutMaxSizes[2], BATCH / COPY_TESNOR_NR};
    unsigned copiedReluMinSizes[] = {C, maxpoolOutMinSizes[1], maxpoolOutMinSizes[2], BATCH / COPY_TESNOR_NR};

    std::vector<unsigned> copyTensorsInput;
    for (unsigned i = 0; i < COPY_TESNOR_NR; ++i)
    {
        unsigned tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, copiedReluMaxSizes,
                                       TENSOR_DIMS, syn_type_single, nullptr, copiedReluMinSizes);
        copyTensorsInput.push_back(tensor);
    }

    unsigned splitDim = SPLIT_DIM;
    addNodeToGraph(NodeFactory::splitNodeTypeName, {splitInTensor}, copyTensorsInput, (void*)&splitDim, sizeof(unsigned));

    std::vector<unsigned> concatTensorsInput;
    for (unsigned i = 0; i < COPY_TESNOR_NR; ++i)
    {
        unsigned tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, copiedReluMaxSizes,
                                       TENSOR_DIMS, syn_type_single, nullptr, copiedReluMinSizes);

        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {copyTensorsInput[i]}, {tensor});
        concatTensorsInput.push_back(tensor);
    }

    unsigned outputTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxpoolOutMaxSizes,
                                                TENSOR_DIMS, syn_type_single, nullptr, nullptr,
                                                0, 0, nullptr,  maxpoolOutMinSizes);

    addNodeToGraph(NodeFactory::concatenateNodeTypeName, concatTensorsInput, {outputTensor}, &splitDim, sizeof(unsigned));

    compileTopology();

    unsigned inActualSizes[] = {C, inActualW, inActualH, BATCH};
    setActualSizes(inTensor, inActualSizes);
    setActualSizes(biasShape, maxpoolOutActualSizes);
    setActualSizes(outputTensor, maxpoolOutActualSizes);
    runTopology(0, true);

    float* weightsPart1Data = castHostInBuffer<float>(weightsPart1);
    float* weightsPart2Data = castHostInBuffer<float>(weightsPart2);
    size_t weightsSize = convWeightsSize[0] * convWeightsSize[1] * convWeightsSize[2] * convWeightsSize[3];

    std::unique_ptr<float[]> refWeights(new float[weightsSize]);

    for (int i = 0; i < weightsSize; i++)
    {
        refWeights[i] = weightsPart1Data[i] + weightsPart2Data[i];
    }

    auto inDesc      = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    const auto weightDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(weightsFinal));
    auto convOutDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(maxpoolInTensor));
    float* inData = castHostInBuffer<float>(inTensor);

    size_t convSize = convOutMaxSizes[0] * convOutMaxSizes[1] * convOutMaxSizes[2] * convOutMaxSizes[3];
    std::unique_ptr<float[]> refConv(new float[convSize]);

    memcpy(inDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);
    memcpy(convOutDesc.m_sizes, convOutActualSizes, sizeof(unsigned) * TENSOR_DIMS);

    calculateFwdConvolution(inDesc,
                            (char*)inData,
                            weightDesc,
                            (char*)refWeights.get(),
                            convOutDesc,
                            (char*)refConv.get(),
                            convParams,
                            m_deviceType);

    float mean[C];
    float iStd[C];
    float runningMean[C];
    float runningVar[C];
    calcBatchNormForwardRef<float>(mean, iStd, runningMean, runningVar, refConv.get(), bnParams.momentum,
                                   refConv.get(), castHostInBuffer<float>(bnBetaTensor),
                                   castHostInBuffer<float>(bnGammaTensor), convOutActualSizes);

    size_t maxpoolSize = maxpoolOutMaxSizes[0] * maxpoolOutMaxSizes[1] * maxpoolOutMaxSizes[2] * maxpoolOutMaxSizes[3];
    std::unique_ptr<float[]> refOut(new float[maxpoolSize]);
    memset(refOut.get(), 0, maxpoolSize * sizeof(float));
    CalcMaxpool2D_2by2(refConv.get(), convOutActualSizes, TENSOR_DIMS, refOut.get());

    float* broadcastData = castHostInBuffer<float>(biasTensor);
    for (int i = 0; i < maxpoolSize; i++)
    {
        refOut[i] += broadcastData[i % C];
        refOut[i] = refOut[i] > 0 ? refOut[i] : 0;
    }

    unsigned actualElements = maxpoolOutActualSizes[0] * maxpoolOutActualSizes[1] *
                              maxpoolOutActualSizes[2] * maxpoolOutActualSizes[3];
    float* out = castHostOutBuffer<float>(outputTensor);
    bool testPass = true;
    for (int i = 0; i < actualElements; i++)
    {
        if (!float_eq(refOut[i], out[i], 0.1))
        {
            testPass = false;
            LOG_ERR(SYN_TEST, "index = {}, cpu = {}, device = {}", i, refOut[i], out[i]);
        }
    }
    for (int i = actualElements; i < maxpoolSize; i++)
    {
        if (!float_eq(0.0, out[i], 0.1))
        {
            testPass = false;
            LOG_ERR(SYN_TEST, "index = {}, cpu = {}, device = {}", i, refOut[i], out[i]);
        }
    }
    ASSERT_TRUE(testPass);
}

TEST_P_GC(SynGaudiDynamicShapesMilestone2, ms3_test, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    if (GetParam())
    {
        HB_ASSERT(GCFG_ENABLE_STAGED_SUBMISSION.value() == true, "ENABLE_STAGED_SUBMISSION should be true for this test variant");
    }
    ScopedConfigurationChange enableInternalNodes("ENABLE_INTERNAL_NODES", "true");
    size_t inActualW = 100;
    size_t inActualH = 92;

    synConvolutionParams convParams;
    convParams.dH   = 1;
    convParams.dW   = 1;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.dilH = 1;
    convParams.dilW = 1;
    convParams.padT = 0;
    convParams.padB = 0;
    convParams.padL = 0;
    convParams.padR = 0;

    // Create the static part of the graph that is inserted as bias to the conv.
    unsigned convBiasSize[] = {C};

    unsigned biasPart1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                             convBiasSize, 1, syn_type_float);

    unsigned copiedBiasPart1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                               convBiasSize, 1, syn_type_float);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {biasPart1}, {copiedBiasPart1});

    unsigned biasPart2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                             convBiasSize, 1, syn_type_float);

    unsigned copiedBiasPart2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                            convBiasSize, 1, syn_type_float);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {biasPart2}, {copiedBiasPart2});

    unsigned biasFinal = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                       convBiasSize, 1, syn_type_float);

    addNodeToGraph("add_fwd_f32", {copiedBiasPart1, copiedBiasPart2}, {biasFinal});

    // Create main branch
    unsigned inMaxSizes[] = {C, MAX_W, MAX_H, BATCH};
    unsigned inMinSizes[] = {C, MIN_W, MIN_H, BATCH};
    unsigned convWeightsSize[] = {C, C, convParams.kW, convParams.kH};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, inMinSizes);

    unsigned convWeightsTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                              convWeightsSize, TENSOR_DIMS, syn_type_float, nullptr);

    const unsigned convOutMaxW = convOutputDimSize(MAX_W, convParams.kW, convParams.dW,
                                                   convParams.padL + convParams.padR, convParams.dilW);
    const unsigned convOutMinW = convOutputDimSize(MIN_W, convParams.kW, convParams.dW,
                                                   convParams.padL + convParams.padR, convParams.dilW);
    const unsigned convOutActualW = convOutputDimSize(inActualW, convParams.kH, convParams.dH,
                                                      convParams.padT + convParams.padB, convParams.dilH);
    const unsigned convOutMaxH = convOutputDimSize(MAX_H, convParams.kH, convParams.dH,
                                                   convParams.padT + convParams.padB, convParams.dilH);
    const unsigned convOutMinH = convOutputDimSize(MIN_H, convParams.kH, convParams.dH,
                                                   convParams.padT + convParams.padB, convParams.dilH);
    const unsigned convOutActualH = convOutputDimSize(inActualH, convParams.kH, convParams.dH,
                                                      convParams.padT + convParams.padB, convParams.dilH);

    unsigned convOutMaxSizes[] = {C, convOutMaxW, convOutMaxH, BATCH};
    unsigned convOutMinSizes[] = {C, convOutMinW, convOutMinH, BATCH};
    unsigned convOutActualSizes[] = {C, convOutActualW, convOutActualH, BATCH};

    unsigned bnInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                       convOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, convOutMinSizes);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {inTensor, convWeightsTensor, biasFinal},
                   {bnInTensor}, (void*)&convParams, sizeof(synConvolutionParams));

    unsigned bnTensorSizes[] = {C};

    unsigned bnBetaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                bnTensorSizes, 1, syn_type_float, nullptr, "Beta");
    unsigned bnGammaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                 bnTensorSizes, 1, syn_type_float, nullptr, "Gamma");
    unsigned bnInMeanTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                  bnTensorSizes, 1, syn_type_float, nullptr, "In Mean");
    unsigned bnInVarTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                 bnTensorSizes, 1, syn_type_float, nullptr, "In Var");

    unsigned maxpoolInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        convOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, convOutMinSizes);

    unsigned bnOutMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                            bnTensorSizes, 1, syn_type_float);
    unsigned bnOutStdTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                           bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                               bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunVarTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                              bnTensorSizes, 1, syn_type_float);

    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;

    addNodeToGraph("batch_norm_fwd_f32", {bnInTensor, bnBetaTensor, bnGammaTensor, bnInMeanTensor, bnInVarTensor},
                   {maxpoolInTensor, bnOutMeanTensor, bnOutStdTensor, bnOutRunMeanTensor, bnOutRunVarTensor},
                   &bnParams, sizeof(ns_BatchNormKernel::Params));

    ns_SpatialReduction::Params kernel_params;
    kernel_params.pad_w_begin = 0;
    kernel_params.pad_h_end   = 0;
    kernel_params.pad_w_end   = 0;
    kernel_params.pad_h_begin = 0;
    kernel_params.kernel_w    = 2;
    kernel_params.kernel_h    = 2;
    kernel_params.stride_w    = 2;
    kernel_params.stride_h    = 2;
    kernel_params.dilation_w  = 1;
    kernel_params.dilation_h  = 1;
    kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    unsigned poolOutMaxW = (convOutMaxW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                           kernel_params.stride_w + 1;
    unsigned poolOutMinW = (convOutMinW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                           kernel_params.stride_w + 1;
    unsigned poolOutActualW = (convOutActualW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                              kernel_params.stride_w + 1;

    unsigned poolOutMaxH = (convOutMaxH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1) /
                           kernel_params.stride_h + 1;
    unsigned poolOutMinH = (convOutMinH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1) /
                           kernel_params.stride_h + 1;
    unsigned poolOutActualH = (convOutActualH + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) /
                              kernel_params.stride_w + 1;

    unsigned maxpoolOutMaxSizes[] = {C, poolOutMaxW, poolOutMaxH, BATCH};
    unsigned maxpoolOutMinSizes[] = {C, poolOutMinW, poolOutMinH, BATCH};
    unsigned maxpoolOutActualSizes[] = {C, poolOutActualW, poolOutActualH, BATCH};

    unsigned maxpoolRetainTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                maxpoolOutMaxSizes, TENSOR_DIMS, syn_type_uint8, nullptr, maxpoolOutMinSizes);

    unsigned reshapeInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                            maxpoolOutMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, maxpoolOutMinSizes);

    addNodeToGraph("maxpool_2d_fwd_f32", {maxpoolInTensor},
                   {maxpoolRetainTensor, reshapeInTensor}, &kernel_params, sizeof(ns_SpatialReduction::Params));

    unsigned reshapeOutMaxSizes[] = {maxpoolOutMaxSizes[0],
                                     maxpoolOutMaxSizes[1] * maxpoolOutMaxSizes[2] * maxpoolOutMaxSizes[3]};
    unsigned reshapeOutMinSizes[] = {maxpoolOutMinSizes[0],
                                     maxpoolOutMinSizes[1] * maxpoolOutMinSizes[2] * maxpoolOutMinSizes[3]};
    unsigned reshapeOutActualSizes[] = {maxpoolOutActualSizes[0],
                                        maxpoolOutActualSizes[1] * maxpoolOutActualSizes[2] * maxpoolOutActualSizes[3]};
    unsigned reshapeTensorDim = 2;

    unsigned reshapeShapeTensor = createShapeTensor(INPUT_TENSOR, reshapeOutMaxSizes, reshapeOutMinSizes, reshapeTensorDim);
    unsigned addInTensor = createTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        reshapeOutMaxSizes, reshapeTensorDim, syn_type_float, nullptr, reshapeOutMinSizes);

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {reshapeInTensor, reshapeShapeTensor}, {addInTensor});

    unsigned biasSizes[] = {C, 1, 1, 1};
    unsigned biasTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr,
                                              biasSizes, reshapeTensorDim, syn_type_float,nullptr, "Bias");

    unsigned biasShape = createShapeTensor(INPUT_TENSOR, reshapeOutMaxSizes, reshapeOutMinSizes, reshapeTensorDim,
                                           syn_type_float, "Broadcast Size", 0);

    unsigned broadcastOutTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                               reshapeOutMaxSizes, reshapeTensorDim, syn_type_float, nullptr, reshapeOutMinSizes);

    addNodeToGraph(NodeFactory::broadcastNodeTypeName, {biasTensor, biasShape}, {broadcastOutTensor});

    unsigned reluInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                         reshapeOutMaxSizes, reshapeTensorDim, syn_type_float, nullptr, reshapeOutMinSizes);

    addNodeToGraph("add_fwd_f32", {addInTensor, broadcastOutTensor}, {reluInTensor});


    unsigned firstTransposeInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                          reshapeOutMaxSizes, reshapeTensorDim, syn_type_float, nullptr, reshapeOutMinSizes);

    addNodeToGraph("relu_fwd_f32", {reluInTensor}, {firstTransposeInTensor});

    unsigned transposeMaxSizes[] = {reshapeOutMaxSizes[1], reshapeOutMaxSizes[0]};
    unsigned transposeMinSizes[] = {reshapeOutMinSizes[1], reshapeOutMinSizes[0]};
    unsigned splitInTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                          transposeMaxSizes, reshapeTensorDim, syn_type_float, nullptr, transposeMinSizes);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2};
    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {firstTransposeInTensor},
                   {splitInTensor},
                   &transposeParams, sizeof (transposeParams));

    unsigned copiedReluMaxSizes[] = {transposeMaxSizes[0], transposeMaxSizes[1] / COPY_TESNOR_NR};
    unsigned copiedReluMinSizes[] = {transposeMinSizes[0], transposeMinSizes[1] / COPY_TESNOR_NR};
    unsigned copiedTransposedReluMaxSizes[] = {copiedReluMaxSizes[1], copiedReluMaxSizes[0]};
    unsigned copiedTransposedReluMinSizes[] = {copiedReluMinSizes[1], copiedReluMinSizes[0]};
    unsigned copiedTransposedReluActualSizes[] = {reshapeOutActualSizes[0] / COPY_TESNOR_NR, reshapeOutActualSizes[1]};

    std::vector<unsigned> copyTensorsInput;
    for (unsigned i = 0; i < COPY_TESNOR_NR; ++i)
    {
        unsigned tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, copiedReluMaxSizes,
                                       reshapeTensorDim, syn_type_single, nullptr, copiedReluMinSizes);
        copyTensorsInput.push_back(tensor);
    }

    unsigned splitDim = reshapeTensorDim - 1;
    addNodeToGraph(NodeFactory::splitNodeTypeName, {splitInTensor}, copyTensorsInput, (void*)&splitDim, sizeof(unsigned));

    std::vector<unsigned> concatTensorsInput;
    for (unsigned i = 0; i < COPY_TESNOR_NR; ++i)
    {
        unsigned tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, copiedReluMaxSizes,
                                       reshapeTensorDim, syn_type_single, nullptr, copiedReluMinSizes);

        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {copyTensorsInput[i]}, {tensor});

        unsigned transposedTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, copiedTransposedReluMaxSizes,
                                                 reshapeTensorDim, syn_type_single, nullptr, copiedTransposedReluMinSizes);

        addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {tensor},
                   {transposedTensor},
                   &transposeParams, sizeof (transposeParams));

        unsigned constTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, copiedTransposedReluMaxSizes,
                                            reshapeTensorDim, syn_type_single, nullptr, copiedTransposedReluMinSizes);
        unsigned constShapeTensor = createShapeTensor(INPUT_TENSOR, copiedTransposedReluMaxSizes, copiedTransposedReluMinSizes,
                                                 reshapeTensorDim, syn_type_uint32);
        ns_ConstantKernel::Params constParams;
        constParams.constant.f = 0.3;

        addNodeToGraph("constant_f32", {constShapeTensor}, {constTensor}, &constParams, sizeof(ns_ConstantKernel::Params));

        // Set it now instead of keep track of these tensors.
        setActualSizes(constShapeTensor, copiedTransposedReluActualSizes);

        unsigned branchOutTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, copiedTransposedReluMaxSizes,
                                                reshapeTensorDim, syn_type_single, nullptr, copiedTransposedReluMinSizes);

        addNodeToGraph("add_fwd_f32", {transposedTensor, constTensor}, {branchOutTensor});
        concatTensorsInput.push_back(branchOutTensor);
    }

    unsigned concatOutputTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, reshapeOutMaxSizes,
                                                reshapeTensorDim, syn_type_single, nullptr, reshapeOutMinSizes);

    unsigned concatDim = 0;
    addNodeToGraph(NodeFactory::concatenateNodeTypeName, concatTensorsInput, {concatOutputTensor}, &concatDim, sizeof(unsigned));

    unsigned reshapeBackShapeTensor = createShapeTensor(INPUT_TENSOR, maxpoolOutMaxSizes, maxpoolOutMinSizes, TENSOR_DIMS);
    unsigned outputTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxpoolOutMaxSizes,
                                                TENSOR_DIMS, syn_type_single, nullptr, nullptr,
                                                0, 0, nullptr,  maxpoolOutMinSizes);

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {concatOutputTensor, reshapeBackShapeTensor},
                   {outputTensor});
    compileTopology();

    unsigned inActualSizes[] = {C, inActualW, inActualH, BATCH};
    setActualSizes(inTensor, inActualSizes);
    setActualSizes(reshapeShapeTensor, reshapeOutActualSizes);
    setActualSizes(biasShape, reshapeOutActualSizes);
    setActualSizes(reshapeBackShapeTensor, maxpoolOutActualSizes);
    setActualSizes(outputTensor, maxpoolOutActualSizes);
    runTopology(0, true);

    float* biasPart1Data = castHostInBuffer<float>(biasPart1);
    float* biasPart2Data = castHostInBuffer<float>(biasPart2);

    size_t biasSize = biasSizes[0];

    std::unique_ptr<float[]> refBias(new float[biasSize]);

    for (int i = 0; i < biasSize; i++)
    {
        refBias[i] = biasPart1Data[i] + biasPart2Data[i];
    }

    auto inDesc      = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    const auto weightDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(convWeightsTensor));
    auto convOutDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(maxpoolInTensor));
    float* inData = castHostInBuffer<float>(inTensor);
    float* weights = castHostInBuffer<float>(convWeightsTensor);

    size_t convSize = convOutMaxSizes[0] * convOutMaxSizes[1] * convOutMaxSizes[2] * convOutMaxSizes[3];
    std::unique_ptr<float[]> refConv(new float[convSize]);

    memcpy(inDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);
    memcpy(convOutDesc.m_sizes, convOutActualSizes, sizeof(unsigned) * TENSOR_DIMS);

    calculateFwdConvolution(inDesc,
                            (char*)inData,
                            weightDesc,
                            (char*)weights,
                            convOutDesc,
                            (char*)refConv.get(),
                            convParams,
                            m_deviceType);
    for (int i = 0; i < convSize; i++)
    {
        refConv[i] += refBias[i % biasSize];
    }

    float mean[C];
    float iStd[C];
    float runningMean[C];
    float runningVar[C];
    calcBatchNormForwardRef<float>(mean, iStd, runningMean, runningVar, refConv.get(), bnParams.momentum,
                                   refConv.get(), castHostInBuffer<float>(bnBetaTensor),
                                   castHostInBuffer<float>(bnGammaTensor), convOutActualSizes);

    size_t maxpoolSize = maxpoolOutMaxSizes[0] * maxpoolOutMaxSizes[1] * maxpoolOutMaxSizes[2] * maxpoolOutMaxSizes[3];
    std::unique_ptr<float[]> refOut(new float[maxpoolSize]);
    memset(refOut.get(), 0, maxpoolSize * sizeof(float));
    CalcMaxpool2D_2by2(refConv.get(), convOutActualSizes, TENSOR_DIMS, refOut.get());

    float* broadcastData = castHostInBuffer<float>(biasTensor);
    for (int i = 0; i < maxpoolSize; i++)
    {
        refOut[i] += broadcastData[i % C];
        refOut[i] = refOut[i] > 0 ? refOut[i] : 0;
        refOut[i] += 0.3;
    }

    unsigned actualElements = maxpoolOutActualSizes[0] * maxpoolOutActualSizes[1] *
                              maxpoolOutActualSizes[2] * maxpoolOutActualSizes[3];
    float* out = castHostOutBuffer<float>(outputTensor);
    bool testPass = true;
    for (int i = 0; i < actualElements; i++)
    {
        if (!float_eq(refOut[i], out[i], 0.1))
        {
            testPass = false;
            LOG_ERR(SYN_TEST, "index = {}, cpu = {}, device = {}", i, refOut[i], out[i]);
        }
    }
    for (int i = actualElements; i < maxpoolSize; i++)
    {
        if (!float_eq(0.0, out[i], 0.1))
        {
            testPass = false;
            LOG_ERR(SYN_TEST, "index = {}, cpu = {}, device = {}", i, refOut[i], out[i]);
        }
    }
    ASSERT_TRUE(testPass);
}
