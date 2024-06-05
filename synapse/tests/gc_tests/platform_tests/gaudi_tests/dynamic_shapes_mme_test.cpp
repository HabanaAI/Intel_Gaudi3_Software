#include "gc_dynamic_shapes_infra.h"
#include "synapse_common_types.h"

// TODO: add mme memset test.

class SynGaudiDynamicMME : public SynGaudiDynamicShapesTestsInfra,
                           public testing::WithParamInterface<size_t>
{
public:
    const size_t maxBatch = 7;
    const size_t minBatch = 2;
    const size_t batchIndex = 3;
};

class SynGaudiSimpleDynamicConv : public SynGaudiDynamicMME {};

INSTANTIATE_TEST_SUITE_P(, SynGaudiSimpleDynamicConv, ::testing::Values(2, 4, 5, 7));

TEST_P_GC(SynGaudiSimpleDynamicConv, simple_conv_ASIC, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned actualBatch = GetParam();
    ASSERT_TRUE(actualBatch >= minBatch && actualBatch <= maxBatch)
        << fmt::format("invalid actualBatch {} valid range [{},{}]", actualBatch, minBatch, maxBatch);

    const unsigned tensorDim = 4;
    size_t H = 128;
    size_t W = 128;
    size_t C = 16;

    unsigned inMaxSize[] = {C, W, H, maxBatch};
    unsigned inMinSize[] = {C, W, H, minBatch};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    synConvolutionParams params;
    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 3;
    params.kW   = 3;
    params.dilH = 1;
    params.dilW = 1;
    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    unsigned wDimSizes[] = { C, C, params.kW, params.kH };

    const unsigned convW = convOutputDimSize(W, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned convH = convOutputDimSize(H, params.kH, params.dH, params.padT + params.padB, params.dilH);

    unsigned convOutMaxDims[] = {C, convW, convH, maxBatch};
    unsigned convOutMinDims[] = {C, convW, convH, minBatch};
    unsigned convOutActualDims[] = {C, convW, convH, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned wTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                           wDimSizes, tensorDim, syn_type_single);


    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convOutMaxDims,
                                             tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, convOutMinDims);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {inTensor, wTensor}, {outTensor}, (void*)&params, sizeof(synConvolutionParams));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, convOutActualDims);
    runTopology(0, true);

    const auto inDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    const auto wDesc   = static_cast<synTensorDescriptor>(getTensorDescriptor(wTensor));
    const auto outDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(outTensor));
    float* inData = castHostInBuffer<float>(inTensor);
    float* wData = castHostInBuffer<float>(wTensor);
    float* outData = castHostOutBuffer<float>(outTensor);

    size_t outBatchElements = C * convW * convH;
    size_t inBatchElements  = C * W * H;
    memset(&outData[outBatchElements * actualBatch], 0, outBatchElements * (maxBatch - actualBatch) * sizeof(float));
    memset(&inData[inBatchElements * actualBatch], 0, inBatchElements * (maxBatch - actualBatch) * sizeof(float));

    CoordArray wrongIdx = {0};
    float expectedResult = 0;
    bool       ret            = checkMmeOp(inDesc,
                          (char*)inData,
                          wDesc,
                          (char*)wData,
                          outDesc,
                          (char*)outData,
                          params,
                          ERepefenceOp::REFERENCE_OP_FWD,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, outDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, outDesc.m_dataType, outData)
                     << " Expected: " << expectedResult;
}
class SynGaudiDynamicGconv : public SynGaudiTestInfra
{
};

TEST_F_GC(SynGaudiDynamicGconv, simple_grouped_conv_dynamic_vs_static_ASIC_CI, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    GlobalConfTestSetter gConvVar("ENABLE_GCONV_PACKING", "false");
    GlobalConfTestSetter gConvVar2("ENABLE_CONV_PACKING_TRAINING", "false");

    const unsigned tensorDim      = 4;
    const size_t   maxBatch       = 4;
    const size_t   minBatch       = 1;
    const unsigned actualBatch    = 2;
    size_t ifmH = 130;
    size_t ifmW = 130;
    size_t C = 256;
    size_t K = 256;
    unsigned inMaxSize[] = {C, ifmW, ifmH, maxBatch};
    unsigned inMinSize[] = {C, ifmW, ifmH, minBatch};
    unsigned inActualSize[] = {C, ifmW, ifmH, actualBatch};

    synConvolutionParams params;
    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 3;
    params.kW   = 3;
    params.dilH = 1;
    params.dilW = 1;
    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;
    params.nGroups = 32;

    unsigned wDimSizes[] = { K, C / params.nGroups, params.kW, params.kH };
    size_t ofmW = convOutputDimSize(inMaxSize[1], params.kW, params.dW, params.padL + params.padR, params.dilW);
    size_t ofmH = convOutputDimSize(inMaxSize[2], params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned convOutMaxSizes[] = {K, ofmW, ofmH, maxBatch};
    unsigned convOutMinSizes[] = {K, ofmW, ofmH, minBatch};
    unsigned convOutActualSizes[] = {K, ofmW, ofmH, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned wTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                           wDimSizes, tensorDim, syn_type_single);


    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convOutMaxSizes,
                                             tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, convOutMinSizes);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {inTensor, wTensor}, {outTensor}, (void*)&params, sizeof(synConvolutionParams));


    unsigned inStaticTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, castHostInBuffer<float>(inTensor),
                                            inActualSize, tensorDim, syn_type_single);

    unsigned wStaticTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, castHostInBuffer<float>(wTensor),
                                           wDimSizes, tensorDim, syn_type_single);

    unsigned outStaticTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convOutActualSizes,
                                             tensorDim, syn_type_single);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {inStaticTensor, wStaticTensor}, {outStaticTensor}, (void*)&params, sizeof(synConvolutionParams));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, convOutActualSizes);
    runTopology(0, true);

    validateResult(castHostOutBuffer<float>(outStaticTensor),
                   castHostOutBuffer<float>(outTensor),
                   K * ofmW * ofmH * actualBatch);
}

void SynGaudiSimpleDynamicGemmAllDynamicBase::createRecipe(unsigned graphIndex)
{
    unsigned buf_size = std::max(params.op1MaxSizes[0]*params.op1MaxSizes[1],
                                 params.op2MaxSizes[0]*params.op2MaxSizes[1]);
    std::vector<float> buf(buf_size);
    std::iota(buf.begin(), buf.end(), 1);

    params.opA = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, buf.data(),
                                     params.op1MaxSizes, params.tensorDim, syn_type_single, nullptr, nullptr,
                                     graphIndex, 0, nullptr, params.op1MinSizes);

    params.opB = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, buf.data(),
                                     params.op2MaxSizes, params.tensorDim, syn_type_single, nullptr, nullptr,
                                     graphIndex, 0, nullptr, params.op2MinSizes);

    params.output = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        params.outputMaxSizes, params.tensorDim, syn_type_single, nullptr, nullptr,
                                        graphIndex, 0, nullptr, params.outputMinSizes);

    addNodeToGraph(NodeFactory::gemmNodeTypeName, {params.opA, params.opB}, {params.output}, &params.gemmParams, sizeof(params.gemmParams),
                   nullptr/*nodeName*/, graphIndex);

    compileTopology("", graphIndex);

    ASSERT_NE(m_graphs[graphIndex].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[graphIndex].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);
}

void SynGaudiSimpleDynamicGemmAllDynamicBase::runRecipe(unsigned actualBatch, unsigned graphIndex, synStatus expected)
{
    unsigned op1Actual = actualBatch;
    unsigned op2Actual = op1Actual * 2;

    setActualSizes(params.opA,    {params.op1WMax, op1Actual},      graphIndex);
    setActualSizes(params.opB,    {op2Actual,      params.op2HMax}, graphIndex);
    setActualSizes(params.output, {op2Actual,      op1Actual},      graphIndex);
    runTopology(graphIndex, true, expected);
}


void SynGaudiSimpleDynamicGemmAllDynamicBase::checkResults(unsigned actualBatch)
{
    unsigned op1Actual = actualBatch;
    unsigned op2Actual = op1Actual * 2;

    auto aDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(params.opA));
    auto bDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(params.opB));
    auto cDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(params.output));
    float *aData = castHostBuffer<float>(params.opA);
    float *bData = castHostBuffer<float>(params.opB);
    float *cData = castHostBuffer<float>(params.output);

    size_t refOutputLen = op1Actual * op2Actual;
    std::unique_ptr<float[]> refOutput(new float[refOutputLen]);

    aDesc.m_sizes[1] = op1Actual;
    bDesc.m_sizes[0] = op2Actual;
    cDesc.m_sizes[0] = op2Actual;
    cDesc.m_sizes[1] = op1Actual;

    calculateGemm(aDesc,
                  (char*)aData,
                  bDesc,
                  (char*)bData,
                  cDesc,
                  (char*)refOutput.get(),
                  params.gemmParams,
                  ERepefenceOp::REFERENCE_OP_FWD,
                  m_deviceType);

    for (int i = 0; i < refOutputLen; i++)
    {
        ASSERT_TRUE(float_eq(refOutput.get()[i], cData[i], 0.01))
            << "index: " << i << ", exp: " << refOutput[i] << ", actual: " << cData[i];
    }
}

class SynGaudiSimpleDynamicGemmAllDynamic : public SynGaudiSimpleDynamicGemmAllDynamicBase,
                                            public testing::WithParamInterface<std::vector<unsigned>> {};

const std::vector<std::vector<unsigned>> gemmParams{{64},
                                                    {90},
                                                    {103},
                                                    {128},
                                                    {64, 103, 90, 128}};
INSTANTIATE_TEST_SUITE_P(, SynGaudiSimpleDynamicGemmAllDynamic, ::testing::ValuesIn(gemmParams));

TEST_P_GC(SynGaudiSimpleDynamicGemmAllDynamic, gemm_with_rt, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    createRecipe();

    std::vector<unsigned> batches = GetParam();

    for (auto actual : batches)
    {
        runRecipe(actual);
        checkResults(actual);
    }
}

class SynGaudiSimpleDynamicGemmCommonDynamic : public SynGaudiDynamicMME {};

INSTANTIATE_TEST_SUITE_P(, SynGaudiSimpleDynamicGemmCommonDynamic, ::testing::Values(64, 128, 193, 256));

TEST_P_GC(SynGaudiSimpleDynamicGemmCommonDynamic, gemm_with_rt, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned actual      = GetParam();
    unsigned op1WMax     = 256;
    unsigned op1WMin     = 64;
    unsigned op1HMax     = 128;
    unsigned op2WMax     = 512;
    unsigned op2HMax     = 256;
    unsigned op2HMin     = 64;

    size_t tensorDim = 2;
    unsigned op1MaxSizes[] = {op1WMax, op1HMax};
    unsigned op1MinSizes[] = {op1WMin, op1HMax};
    unsigned op2MaxSizes[] = {op2WMax, op2HMax};
    unsigned op2MinSizes[] = {op2WMax, op2HMin};

    unsigned outputMaxSizes[] = {op2WMax, op1HMax};

    unsigned opA = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       op1MaxSizes,
                                       tensorDim,
                                       syn_type_single,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       op1MinSizes);

    unsigned opB = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       op2MaxSizes,
                                       tensorDim,
                                       syn_type_single,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       op2MinSizes);

    unsigned output = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                          outputMaxSizes, tensorDim, syn_type_single, nullptr, nullptr,
                                          0, 0, nullptr, nullptr);

    synGEMMParams gemmParams;
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {opA, opB}, {output}, &gemmParams, sizeof(gemmParams));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(opA, {actual, op1HMax});
    setActualSizes(opB, {op2WMax, actual});
    runTopology(0, false);

    auto aDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(opA));
    auto bDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(opB));
    const auto cDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(output));
    float* aData = castHostBuffer<float>(opA);
    float* bData = castHostBuffer<float>(opB);
    float* cData = castHostBuffer<float>(output);

    size_t refOutputLen = outputMaxSizes[0] * outputMaxSizes[1];
    std::unique_ptr<float[]> refOutput(new float[refOutputLen]);
    aDesc.m_sizes[0] = actual;
    bDesc.m_sizes[1] = actual;

    calculateGemm(aDesc,
                  (char*)aData,
                  bDesc,
                  (char*)bData,
                  cDesc,
                  (char*)refOutput.get(),
                  gemmParams,
                  ERepefenceOp::REFERENCE_OP_FWD,
                  m_deviceType);

    for (int i = 0; i < refOutputLen; i++)
    {
        ASSERT_TRUE(float_eq(refOutput.get()[i], cData[i], 0.01)) << i;
    }
}

class SynGaudiSimpleDynamicGemmDynamicB : public SynGaudiDynamicMME {};

INSTANTIATE_TEST_SUITE_P(, SynGaudiSimpleDynamicGemmDynamicB, ::testing::Values(128, 150, 256));


TEST_P_GC(SynGaudiSimpleDynamicGemmDynamicB, gemm_with_rt, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned op1W    = 256;
    unsigned op1H    = 128;
    unsigned op2WMax = 256;
    unsigned op2WMin = 128;
    unsigned op2H    = 256;

    const unsigned op2Actual = GetParam();
    ASSERT_TRUE(op2Actual >= op2WMin && op2Actual <= op2WMax)
        << fmt::format("invalid actualBatch {} valid range [{},{}]", op2Actual, minBatch, maxBatch);

    size_t tensorDim = 2;
    unsigned op1Sizes[] = {op1W, op1H};
    unsigned op2MaxSizes[] = {op2WMax, op2H};
    unsigned op2MinSizes[] = {op2WMin, op2H};

    unsigned outputMaxSizes[] = {op2WMax, op1H};
    unsigned outputMinSizes[] = {op2WMin, op1H};

    unsigned opA = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                       op1Sizes, tensorDim, syn_type_single, nullptr, nullptr,
                                       0, 0, nullptr, nullptr);

    unsigned opB = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                       op2MaxSizes, tensorDim, syn_type_single, nullptr, nullptr,
                                       0, 0, nullptr, op2MinSizes);

    unsigned output = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                          outputMaxSizes, tensorDim, syn_type_single, nullptr, nullptr,
                                          0, 0, nullptr, outputMinSizes);

    synGEMMParams gemmParams;
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {opA, opB}, {output}, &gemmParams, sizeof(gemmParams));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(opB, {op2Actual, op2H});
    setActualSizes(output, {op2Actual, op1H});
    runTopology(0, true);

    const auto aDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(opA));
    auto bDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(opB));
    auto cDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(output));
    float* aData = castHostBuffer<float>(opA);
    float* bData = castHostBuffer<float>(opB);
    float* cData = castHostBuffer<float>(output);

    size_t refOutputLen = op2Actual * op1H;
    std::unique_ptr<float[]> refOutput(new float[refOutputLen]);

    bDesc.m_sizes[0] = op2Actual;
    cDesc.m_sizes[0] = op2Actual;

    calculateGemm(aDesc,
                  (char*)aData,
                  bDesc,
                  (char*)bData,
                  cDesc,
                  (char*)refOutput.get(),
                  gemmParams,
                  ERepefenceOp::REFERENCE_OP_FWD,
                  m_deviceType);

    for (int i = 0; i < refOutputLen; i++)
    {
        ASSERT_TRUE(float_eq(refOutput.get()[i], cData[i], 0.01)) << i;
    }
}

class SynGaudiSimpleDynamicDeDw : public SynGaudiDynamicMME {};

INSTANTIATE_TEST_SUITE_P(, SynGaudiSimpleDynamicDeDw, ::testing::Values(2, 4, 7));

TEST_P_GC(SynGaudiSimpleDynamicDeDw, dedw, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned actualBatch = GetParam();
    ASSERT_TRUE(actualBatch >= minBatch && actualBatch <= maxBatch)
        << fmt::format("invalid actualBatch {} valid range [{},{}]", actualBatch, minBatch, maxBatch);

    ScopedConfigurationChange slice_disable("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange pipeline_disable("ENABLE_PIPELINE_MANAGEMENT", "false");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    // TODO [SW-175899]: remove once the ticket is closed
    ScopedConfigurationChange disableMmeBatchConcurrency("ENABLE_MME_CD_CONCURRENCY", "false");
    synConvolutionParams params;
    params.kH   = 3;
    params.kW   = 3;

    const unsigned tensorDim = 4;
    size_t xH = 128;
    size_t xW = 128;
    size_t xC = 16;
    size_t yH = convOutputDimSize(xH, params.kH, params.dH, params.padT + params.padB, params.dilH);
    size_t yW = convOutputDimSize(xW, params.kW, params.dW, params.padL + params.padR, params.dilW);
    size_t yC = xC;

    unsigned xMaxSize[] = {xC, xW, xH, maxBatch};
    unsigned xActualSize[] = {xC, xW, xH, actualBatch};
    unsigned xMinSize[] = {xC, xW, xH, minBatch};
    unsigned wSizes[] = { xC, xC, params.kW, params.kH };
    unsigned yMaxSize[] = {yC, yW, yH, maxBatch};
    unsigned yActualSize[] = {yC, yW, yH, actualBatch};
    unsigned yMinSize[] = {yC, yW, yH, minBatch};

    unsigned dedy = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                        yMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                        0, 0, nullptr, yMinSize);

    unsigned x = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                     xMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                     0, 0, nullptr, xMinSize);

    unsigned dedw = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        wSizes, tensorDim, syn_type_single);

    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dedy, x}, {dedw}, &params, sizeof(params));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(x, xActualSize);
    setActualSizes(dedy, yActualSize);

    runTopology(0, true);

    const auto yDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(dedy));
    const auto xDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(x));
    const auto wDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(dedw));
    float* yData = castHostBuffer<float>(dedy);
    float* xData = castHostBuffer<float>(x);
    float* wData = castHostBuffer<float>(dedw);

    size_t yBatchElements = yMaxSize[0] * yMaxSize[1] * yMaxSize[2];
    size_t xBatchElements = xMaxSize[0] * xMaxSize[1] * xMaxSize[2];

    memset(&yData[yBatchElements * actualBatch], 0, yBatchElements * (maxBatch - actualBatch) * sizeof(float));
    memset(&xData[xBatchElements * actualBatch], 0, xBatchElements * (maxBatch - actualBatch) * sizeof(float));

    CoordArray wrongIdx = {0};
    float expectedResult = 0;
    bool       ret            = checkMmeOp(xDesc,
                          (char*)xData,
                          wDesc,
                          (char*)wData,
                          yDesc,
                          (char*)yData,
                          params,
                          ERepefenceOp::REFERENCE_OP_DEDW,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, wDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, wDesc.m_dataType, wData)
                     << " Expected: " << expectedResult;
}

class SynGaudiSimpleDynamicDeDx : public SynGaudiDynamicMME {};

INSTANTIATE_TEST_SUITE_P(SimpleDynamicDeDx_ASIC, SynGaudiSimpleDynamicDeDx, ::testing::Values(2, 4, 7));
INSTANTIATE_TEST_SUITE_P(SimpleDynamicDeDx_ASIC_CI, SynGaudiSimpleDynamicDeDx, ::testing::Values(7));

TEST_P_GC(SynGaudiSimpleDynamicDeDx, dedx_with_shape_tensor_ASIC) // TODO add Gaudi3 when ASIC is ready [SW-157047]
{
    synConvolutionParams params;
    params.kH   = 3;
    params.kW   = 3;

    const unsigned actualBatch = GetParam();
    ASSERT_TRUE(actualBatch >= minBatch && actualBatch <= maxBatch)
        << fmt::format("invalid actualBatch {} valid range [{},{}]", actualBatch, minBatch, maxBatch);

    const unsigned tensorDim = 4;
    size_t xH = 128;
    size_t xW = 128;
    size_t xC = 16;
    size_t yH = convOutputDimSize(xH, params.kH, params.dH, params.padT + params.padB, params.dilH);
    size_t yW = convOutputDimSize(xW, params.kW, params.dW, params.padL + params.padR, params.dilW);
    size_t yC = xC;

    unsigned xMaxSize[] = {xC, xW, xH, maxBatch};
    unsigned xMinSize[] = {xC, xW, xH, minBatch};
    unsigned wSizes[] = { xC, xC, params.kW, params.kH };
    unsigned yMaxSize[] = {yC, yW, yH, maxBatch};
    unsigned yMinSize[] = {yC, yW, yH, minBatch};

    unsigned dedy = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                        yMaxSize, tensorDim, syn_type_single, nullptr, "dy",
                                        0, 0, nullptr, yMinSize);

    unsigned w = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                     wSizes, tensorDim, syn_type_single, nullptr, "w");

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        xMaxSize, tensorDim, syn_type_single, nullptr, "dx",
                                        0, 0, nullptr, xMinSize);

    unsigned dxShape = createShapeTensor(INPUT_TENSOR, xMaxSize, xMinSize, tensorDim, syn_type_single, "dx_shape", 0);

    addNodeToGraph(NodeFactory::deDxNodeTypeName, {dedy, w, dxShape}, {dedx}, &params, sizeof(params));

    compileTopology();
    ASSERT_FALSE(HasFailure());

    setActualSizes(dedy, {yC, yW, yH, actualBatch});
    setActualSizes(dedx, {xC, xW, xH, actualBatch});
    setActualSizes(dxShape, {xC, xW, xH, actualBatch});

    runTopology(0, true);

    const auto yDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(dedy));
    const auto xDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(dedx));
    const auto wDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(w));
    float* yData = castHostBuffer<float>(dedy);
    float* xData = castHostBuffer<float>(dedx);
    float* wData = castHostBuffer<float>(w);

    size_t yBatchElements = yMaxSize[0] * yMaxSize[1] * yMaxSize[2];
    memset(&yData[yBatchElements * actualBatch], 0, yBatchElements * (maxBatch - actualBatch) * sizeof(float));

    CoordArray wrongIdx = {0};
    float expectedResult = 0;
    bool       ret            = checkMmeOp(xDesc,
                          (char*)xData,
                          wDesc,
                          (char*)wData,
                          yDesc,
                          (char*)yData,
                          params,
                          ERepefenceOp::REFERENCE_OP_DEDX,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, wDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, wDesc.m_dataType, wData)
                     << " Expected: " << expectedResult;

    // verify that the output is calculated only for the actual batch range
    size_t xMaxElements = xMaxSize[0] * xMaxSize[1] * xMaxSize[2] * xMaxSize[3];
    size_t xActualElement = xMaxSize[0] * xMaxSize[1] * xMaxSize[2] * actualBatch;

    for(size_t i = xActualElement; i < xMaxElements; ++i)
    {
        ASSERT_EQ(0, xData[i]) << "Wrong value at index: " << i << " value: " << xData[i] << " expected: " << 0;
    }
}

TEST_F_GC(SynGaudiSimpleDynamicDeDx, static_flattened_dedx_with_shape_tensor)
{
    synConvolutionParams convParams = {};
    convParams.kW = 1;
    convParams.kH = 1;
    convParams.dW = 1;
    convParams.dH = 1;
    convParams.dilW = 1;
    convParams.dilH = 1;
    convParams.activation.numChannels = 1;
    convParams.nGroups = 1;

    const unsigned tensorDim = 4;

    unsigned xMaxSize[] = { 256, 28, 28, 512 };
    unsigned wSizes[] = { 81, 256, 1, 1 };
    unsigned yMaxSize[] = { 81, 28, 28, 512 };

    unsigned dedy = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                        yMaxSize, tensorDim, syn_type_single, nullptr, "dy",
                                        0, 0, nullptr);

    unsigned w = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                     wSizes, tensorDim, syn_type_single, nullptr, "w");

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        xMaxSize, tensorDim, syn_type_single, nullptr, "dx",
                                        0, 0, nullptr);

    unsigned dxShape = createShapeTensor(INPUT_TENSOR, xMaxSize, xMaxSize, tensorDim, syn_type_single, "dx_shape", 0);

    addNodeToGraph(NodeFactory::deDxNodeTypeName, {dedy, w, dxShape}, {dedx}, &convParams, sizeof(convParams));

    compileTopology();
}

//A reproducer for [SW-31040]
TEST_F_GC(SynGaudiDynamicMME, dedx_with_shape_tensor_ASIC) // TODO add Gaudi3 whern ASIC is ready [SW-157047]
{
    synConvolutionParams params;
    params.kH   = 3;
    params.kW   = 3;
    params.padT = 1;
    params.padB = 1;
    params.padL = 1;
    params.padR = 1;
    const unsigned tensorDim = 4;
    size_t xH = 128;
    size_t xW = 128;
    size_t xC = 256;
    size_t yH = convOutputDimSize(xH, params.kH, params.dH, params.padT + params.padB, params.dilH);
    size_t yW = convOutputDimSize(xW, params.kW, params.dW, params.padL + params.padR, params.dilW);
    size_t yC = xC;

    unsigned xMaxSize[] = {xC, xW, xH, 2};
    unsigned xMinSize[] = {xC, xW, xH, 2};
    unsigned wSizes[] = { xC, xC, params.kW, params.kH };
    unsigned yMaxSize[] = {yC, yW, yH, 2};
    unsigned yMinSize[] = {yC, yW, yH, 2};

    unsigned dedy = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                        yMaxSize, tensorDim, syn_type_single, nullptr, "dy",
                                        0, 0, nullptr, yMinSize);

    unsigned w = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                     wSizes, tensorDim, syn_type_single, nullptr, "w");

    unsigned dedx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                        xMaxSize, tensorDim, syn_type_single, nullptr, "dx",
                                        0, 0, nullptr, xMinSize);

    unsigned dxShape = createShapeTensor(INPUT_TENSOR, xMaxSize, xMinSize, tensorDim, syn_type_single, "dx_shape", 0);

    addNodeToGraph(NodeFactory::deDxNodeTypeName, {dedy, w, dxShape}, {dedx}, &params, sizeof(params));

    compileTopology();
    ASSERT_FALSE(HasFailure());

    setActualSizes(dedy, {yC, yW, yH, 2});
    setActualSizes(dedx, {xC, xW, xH, 2});
    setActualSizes(dxShape, {xC, xW, xH, 2});

    runTopology(0, true);

    const auto yDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(dedy));
    const auto xDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(dedx));
    const auto wDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(w));
    float* yData = castHostBuffer<float>(dedy);
    float* xData = castHostBuffer<float>(dedx);
    float* wData = castHostBuffer<float>(w);

    CoordArray wrongIdx = {0};
    float expectedResult = 0;
    bool       ret            = checkMmeOp(xDesc,
                          (char*)xData,
                          wDesc,
                          (char*)wData,
                          yDesc,
                          (char*)yData,
                          params,
                          ERepefenceOp::REFERENCE_OP_DEDX,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, wDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, wDesc.m_dataType, wData)
                     << " Expected: " << expectedResult;

}
