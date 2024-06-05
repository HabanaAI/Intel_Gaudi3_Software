#include "sram_slicing_test.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "scoped_configuration_change.h"

class SynGaudiSRAMSlicingBigInputConvTest : public SynGaudiSRAMSlicingTest,
                                            public testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int>>
                                                                                //height, width, channels, filter, stride, dilation, padBefore
{
public:
    void testBigImageConv(unsigned height, unsigned width, unsigned channels, unsigned filter, unsigned stride, unsigned dilation, unsigned padBefore);
};

void SynGaudiSRAMSlicingTest::testNonCDSlicing(const unsigned hChunks, const unsigned kChunks, const unsigned inCD)
{
    const unsigned chunkSize = 64;
    const unsigned OUT_BHW   = chunkSize * hChunks;
    const unsigned OUT_K     = chunkSize * kChunks;

    std::stringstream slicerCapVal;
    slicerCapVal << 2 * 2 * chunkSize * sizeof(float) * inCD;
    ScopedConfigurationChange slicerConfig("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    unsigned opASizes[] = {inCD, OUT_BHW, 1, 1};
    unsigned opBSizes[] = {OUT_K, inCD, 1, 1};
    unsigned opOutSizes[] = {OUT_K, OUT_BHW, 1, 1};
    unsigned opAIdx   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, opASizes, 4, syn_type_float, nullptr, "opA");
    unsigned opBIdx   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, opBSizes, 4, syn_type_float, nullptr, "opB");
    unsigned opOutIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, syn_type_float, nullptr, "opOut");

    synConvolutionParams params;
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opAIdx, opBIdx}, {opOutIdx}, &params, sizeof(params), "conv");

    compileAndRun();

    synTensorDescriptor opADesc = m_tensorDescs[opAIdx];
    synTensorDescriptor opBDesc = m_tensorDescs[opBIdx];
    synTensorDescriptor opOutDesc = m_tensorDescs[opOutIdx];
    auto opAData = m_hostBuffers[opAIdx];
    auto opBData = m_hostBuffers[opBIdx];
    auto opOutData = m_hostBuffers[opOutIdx];

    CoordArray wrongIdx = {0};
    bool       ret      = checkFwdConvolution(opADesc,
                                   (char*)opAData,
                                   opBDesc,
                                   (char*)opBData,
                                   opOutDesc,
                                   (char*)opOutData,
                                   params,
                                   wrongIdx,
                                   m_deviceType);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, opOutDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        opOutDesc.m_dataType,
                                                        opOutData);
}

void SynGaudiSRAMSlicingTest::testReluConvSlicing(const char* sramCapStr, bool checkIntermediateTensor)
{
    if (sramCapStr)
    {
        synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", sramCapStr);
    }
    synDataType type = syn_type_float;
    const unsigned b = 64, h = 28, w = 28, c = 128;
    const unsigned r = 3, s = 3, k = 128;

    unsigned reluSizes[] = {c, w, h, b};
    unsigned reluIn      = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               reluSizes,
                                               4,
                                               type,
                                               nullptr,
                                               "reluIn");
    unsigned reluOut, opA;
    if (checkIntermediateTensor)
    {
        reluOut = createPersistTensor(OUTPUT_TENSOR,MEM_INIT_ALL_ZERO, nullptr, reluSizes, 4, type, nullptr, "reluOut");
        opA = connectOutputTensorToInputTensor(reluOut);
    }
    else
    {
        reluOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, reluSizes, 4, type);
        opA     = connectOutputTensorToInputTensor(reluOut);
    }
    unsigned opBSizes[] = {k, c, s, r};
    unsigned opB = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, opBSizes, 4, type, nullptr, "Weights");

    unsigned opOutSizes[] = {k, w, h, b};
    unsigned opOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, type, nullptr, "OpOut");

    synConvolutionParams convParams;
    convParams.kH = 3; convParams.kW = 3;
    convParams.padL = 1; convParams.padR = 1;
    convParams.padT = 1; convParams.padB = 1;
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opA, opB}, {opOut}, &convParams, sizeof(convParams));
    addNodeToGraph((type==syn_type_float) ? "relu_fwd_f32" : "relu_fwd_bf16", {reluIn}, {reluOut});

    addConfigurationToRun(FIRST_RUN,
                          "SRAM_SLICER_MAX_CAPACITY_BYTES",
                          GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.getValueStr());

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    if (checkIntermediateTensor)
    {
        compareRunsResults({reluOut, opOut});
    }
    else
    {
        compareRunsResults({opOut});
    }
}

void SynGaudiSRAMSlicingTest::testReluConvWithEvictionSlicing(const char* sramCapStr)
{
    std::shared_ptr<ScopedConfigurationChange> sramCapacity;

    if (sramCapStr)
    {
        sramCapacity = std::make_shared<ScopedConfigurationChange>("SRAM_SLICER_MAX_CAPACITY_BYTES", sramCapStr);
    }

    synDataType type = syn_type_float;
    const unsigned b = 64, h = 28, w = 28, c = 128;
    const unsigned r = 3, s = 3,  k = 128;

    unsigned reluSizes[] = {c, w, h, b};
    unsigned reluIn      = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          reluSizes,
                                          4,
                                          type,
                                          nullptr,
                                          "reluIn");
    unsigned reluOut     = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, reluSizes, 4, type);
    unsigned opA = connectOutputTensorToInputTensor(reluOut);
    unsigned memCpyOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, reluSizes, 4, type, nullptr, "memCpyOut");
    unsigned opBSizes[] = {k, c, s, r};
    unsigned opB = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, opBSizes, 4, type, nullptr, "weights");

    unsigned opOutSizes[] = {k, w, h, b};
    unsigned opOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, type, nullptr, "convOutput");

    synConvolutionParams convParams;
    convParams.kH = 3; convParams.kW = 3;
    convParams.padL = 1; convParams.padR = 1;
    convParams.padT = 1; convParams.padB = 1;
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opA, opB}, {opOut}, &convParams, sizeof(convParams));
    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut});
    addNodeToGraph("memcpy", {opA} , {memCpyOut});
    compileAndRun();

    const synTensorDescriptor& reluInDesc = m_tensorDescs[reluIn];
    const synTensorDescriptor& reluOutDesc = m_tensorDescs[reluOut];
    const synTensorDescriptor& opBDesc = m_tensorDescs[opB];
    const synTensorDescriptor& opOutDesc = m_tensorDescs[opOut];

    uint64_t           reluOutSize = getNumberOfElements(reluInDesc.m_sizes, reluInDesc.m_dims);
    std::vector<float> calcReluOut(reluOutSize);
    calculateRelu(reluInDesc, m_hostBuffers[reluIn],
                  reluInDesc, calcReluOut.data());

    CoordArray wrongIdx = {0};
    bool ret = checkResults(reluOutDesc, (char*)m_hostBuffers[memCpyOut], (char*)calcReluOut.data(), wrongIdx);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, reluOutDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "RELU: Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " calculated value: " << getIndexValue(sizes, wrongIdx,
                                                               reluOutDesc.m_dataType, calcReluOut.data())
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        reluOutDesc.m_dataType, m_hostBuffers[reluOut]);

    ret = checkFwdConvolution(reluOutDesc,
                              (char*)calcReluOut.data(),
                              opBDesc,
                              (char*)m_hostBuffers[opB],
                              opOutDesc,
                              (char*)m_hostBuffers[opOut],
                              convParams,
                              wrongIdx,
                              m_deviceType);
    castNcopy(sizes, opOutDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "CONV: Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        opOutDesc.m_dataType, m_hostBuffers[opOut]);
}

void SynGaudiSRAMSlicingConsistency::testReluConvWithEvictionSlicingConsistency(const char* sramCapStr)
{
    constexpr unsigned numOfGraphs = 2;

    for (unsigned graphIndex = 1; graphIndex < numOfGraphs; graphIndex++)
    {
        // The first graph already exists
        createGraph();
    }

    std::string topologyName = "testReluConvWithEvictionSlicingTwice";

    std::shared_ptr<ScopedConfigurationChange> sramCapacity;
    if (sramCapStr)
    {
        sramCapacity = std::make_shared<ScopedConfigurationChange>("SRAM_SLICER_MAX_CAPACITY_BYTES", sramCapStr);
    }

    constexpr synDataType type = syn_type_float;
    constexpr unsigned    b = 64, h = 28, w = 28, c = 128;
    constexpr unsigned    r = 3, s = 3, k = 128;
    unsigned       reluSizes[]  = {c, w, h, b};
    unsigned       opBSizes[]   = {k, c, s, r};
    unsigned       opOutSizes[] = {k, w, h, b};

    synConvolutionParams convParams;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.padL = 1;
    convParams.padR = 1;
    convParams.padT = 1;
    convParams.padB = 1;

    for (unsigned graphIndex = 0; graphIndex < numOfGraphs; graphIndex++)
    {
        const unsigned reluIn    = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                    nullptr,
                                                    reluSizes,
                                                    4,
                                                    type,
                                                    nullptr,
                                                    "reluIn",
                                                    graphIndex);
        const unsigned reluOut =
            createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, reluSizes, 4, type, nullptr, nullptr, graphIndex);
        const unsigned opA       = connectOutputTensorToInputTensor(reluOut);
        const unsigned memCpyOut = createPersistTensor(OUTPUT_TENSOR,
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       reluSizes,
                                                       4,
                                                       type,
                                                       nullptr,
                                                       "memCpyOut",
                                                       graphIndex);
        const unsigned opB       = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 opBSizes,
                                                 4,
                                                 type,
                                                 nullptr,
                                                 "weights",
                                                 graphIndex);
        const unsigned opOut     = createPersistTensor(OUTPUT_TENSOR,
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   opOutSizes,
                                                   4,
                                                   type,
                                                   nullptr,
                                                   "convOutput",
                                                   graphIndex);

        addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                       {opA, opB},
                       {opOut},
                       &convParams,
                       sizeof(convParams),
                       nullptr,
                       graphIndex);
        addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, nullptr, graphIndex);
        addNodeToGraph("memcpy", {opA}, {memCpyOut}, nullptr, 0, nullptr, graphIndex);
        compileTopology("gaudi_multiple_graphs_relu_g" + std::to_string(graphIndex), graphIndex);
    }

    const recipe_t& base_recipe = *getGraph(0).recipeHandle->basicRecipeHandle.recipe;

    for (unsigned graphIndex = 1; graphIndex < numOfGraphs; graphIndex++)
    {
        const recipe_t& curr_recipe = *getGraph(graphIndex).recipeHandle->basicRecipeHandle.recipe;
        ASSERT_TRUE(compareRecipes(base_recipe, curr_recipe, true))
            << "recipe comparison failed, recipes are not equal";
    }
}

/* This will test conv->relu-> ... 5 times
 * Tensor sizes are - input - 4kx4k with different batch sizes
 *                    weight - 3x3
 *  Large batch sizes should run only on ASIC to prevent significant CT\RT.
 */
void SynGaudiSRAMSlicingTest::testBigConvReluX5(unsigned batchSize)
{
    unsigned b = batchSize;
    unsigned h = 4096, w = 4096, c = 3, k = 3;
    unsigned xSizes[] = {c, w, h, b};
    unsigned wSizes[] = {k, c, 1 ,1};
    unsigned ySizes[] = {k, w, h, b};

    unsigned opX1Idx    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "X");
    unsigned opWIdx     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes, 4, syn_type_float, nullptr, "Weight");
    unsigned opY1Idx    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opY1InIdx  = connectOutputTensorToInputTensor(opY1Idx);
    unsigned opRY1Idx   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opX2Idx    = connectOutputTensorToInputTensor(opRY1Idx);
    unsigned opY2Idx    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opY2InIdx  = connectOutputTensorToInputTensor(opY2Idx);
    unsigned opRY2Idx   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opX3Idx    = connectOutputTensorToInputTensor(opRY2Idx);
    unsigned opY3Idx    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opY3InIdx  = connectOutputTensorToInputTensor(opY3Idx);
    unsigned opRY3Idx   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opX4Idx    = connectOutputTensorToInputTensor(opRY3Idx);
    unsigned opY4Idx    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opY4InIdx  = connectOutputTensorToInputTensor(opY4Idx);
    unsigned opRY4Idx   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opX5Idx    = connectOutputTensorToInputTensor(opRY4Idx);
    unsigned opY5Idx    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float);
    unsigned opY5InIdx  = connectOutputTensorToInputTensor(opY5Idx);
    unsigned opRY5Idx   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes, 4, syn_type_float, nullptr, "Out");
    synConvolutionParams params;
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opX1Idx, opWIdx}, {opY1Idx}, &params, sizeof(params), "Conv_1");
    addNodeToGraph("relu_fwd_f32", {opY1InIdx}, {opRY1Idx}, nullptr, 0, "Relu_1");

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opX2Idx, opWIdx}, {opY2Idx}, &params, sizeof(params), "Conv_2");
    addNodeToGraph("relu_fwd_f32", {opY2InIdx}, {opRY2Idx}, nullptr, 0, "Relu_2");

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opX3Idx, opWIdx}, {opY3Idx}, &params, sizeof(params), "Conv_3");
    addNodeToGraph("relu_fwd_f32", {opY3InIdx}, {opRY3Idx}, nullptr, 0, "Relu_3");

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opX4Idx, opWIdx}, {opY4Idx}, &params, sizeof(params), "Conv_4");
    addNodeToGraph("relu_fwd_f32", {opY4InIdx}, {opRY4Idx}, nullptr, 0, "Relu_4");

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opX5Idx, opWIdx}, {opY5Idx}, &params, sizeof(params), "Conv_5");
    addNodeToGraph("relu_fwd_f32", {opY5InIdx}, {opRY5Idx}, nullptr, 0, "Relu_5");

    compileAndRun();

    // calculate manually
    const auto xDesc = static_cast<synTensorDescriptor>(m_tensorDescs[opX1Idx]);
    const auto wDesc = static_cast<synTensorDescriptor>(m_tensorDescs[opWIdx]);
    const auto yDesc = static_cast<synTensorDescriptor>(m_tensorDescs[opY1Idx]);

    auto xData = m_hostBuffers[opX1Idx];
    auto wData = m_hostBuffers[opWIdx];
    auto outData = m_hostBuffers[opRY5Idx];
    uint64_t           ySizeInElements = getMemorySize(yDesc.m_sizes, yDesc.m_dataType, yDesc.m_dims) / sizeof(float);
    std::vector<float> calcData1(ySizeInElements), calcData2(ySizeInElements);

    calculateFwdConvolution(xDesc,
                            (char*)xData,
                            wDesc,
                            (char*)wData,
                            yDesc,
                            (char*)calcData1.data(),
                            params,
                            m_deviceType);
    calculateRelu(yDesc, calcData1.data(), yDesc, calcData2.data());
    calculateFwdConvolution(yDesc,
                            (char*)calcData2.data(),
                            wDesc,
                            (char*)wData,
                            yDesc,
                            (char*)calcData1.data(),
                            params,
                            m_deviceType);
    calculateRelu(yDesc, calcData1.data(), yDesc, calcData2.data());
    calculateFwdConvolution(yDesc,
                            (char*)calcData2.data(),
                            wDesc,
                            (char*)wData,
                            yDesc,
                            (char*)calcData1.data(),
                            params,
                            m_deviceType);
    calculateRelu(yDesc, calcData1.data(), yDesc, calcData2.data());
    calculateFwdConvolution(yDesc,
                            (char*)calcData2.data(),
                            wDesc,
                            (char*)wData,
                            yDesc,
                            (char*)calcData1.data(),
                            params,
                            m_deviceType);
    calculateRelu(yDesc, calcData1.data(), yDesc, calcData2.data());
    calculateFwdConvolution(yDesc,
                            (char*)calcData2.data(),
                            wDesc,
                            (char*)wData,
                            yDesc,
                            (char*)calcData1.data(),
                            params,
                            m_deviceType);
    calculateRelu(yDesc, calcData1.data(), yDesc, calcData2.data());

    CoordArray wrongIdx = {0};
    bool ret = checkResults(yDesc, (char*)outData, (char*)calcData2.data(), wrongIdx);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, yDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        yDesc.m_dataType,
                                                        outData);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, validate_basic_bhw_slicing_h_cd_2)
{
    testNonCDSlicing(4, 1, 2);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, validate_basic_bhw_slicing_k_cd_2)
{
    testNonCDSlicing(1, 2, 2);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, validate_basic_bhw_slicing_h_k_cd_2)
{
    testNonCDSlicing(3, 3, 2);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, relu_dedx_smallHWC_L2)
{
    const unsigned b = 5;
    const unsigned r = 3, s = 3;
    const unsigned h = 5, w = 5, c = 5, k = 512;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned wSizes[]       = {k, c, s ,r};
    unsigned dedxOutSizes[] = {c, w + s - 1, h + r - 1 ,b};

    std::stringstream slicerCapVal;
    slicerCapVal << 113920*3 ;
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    synConvolutionParams params;
    params.kH = r;
    params.kW = s;

    unsigned opReluInIdx  = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes, 4, syn_type_float, nullptr, "relu1_in");
    unsigned opReluOutIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, wSizes, 4, syn_type_float, nullptr, "relu1_out");
    unsigned opWIdx       = connectOutputTensorToInputTensor(opReluOutIdx);
    unsigned opDeDyIdx    = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes,    4, syn_type_float, nullptr, "dedy");
    unsigned dedxOutIdx   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, dedxOutSizes, 4, syn_type_float, nullptr, "dedx");
    unsigned relu2InIdx   = connectOutputTensorToInputTensor(dedxOutIdx);
    unsigned relu2OutIdx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, dedxOutSizes, 4, syn_type_float, nullptr, "relu2Out");

    addNodeToGraph("relu_fwd_f32",                {opReluInIdx},       {opReluOutIdx}, nullptr, 0,              "relu_1");
    addNodeToGraph(NodeFactory::deDxNodeTypeName, {opDeDyIdx, opWIdx}, {dedxOutIdx},   &params, sizeof(params), "dedx");
    addNodeToGraph("relu_fwd_f32",                {relu2InIdx},        {relu2OutIdx},  nullptr, 0,              "relu_2");

    compileAndRun();

    unsigned           relu1OutputSize = k*c*s*r;
    float*             relu1Input      = (float *)m_hostBuffers[opReluInIdx];
    std::vector<float> relu1Output(relu1OutputSize);

    for (int i = 0; i < relu1OutputSize ; i++)
    {
        relu1Output[i] = ((relu1Input[i] > 0) ? relu1Input[i] : 0);
    }


    synTensorDescriptor dedyDesc = m_tensorDescs[opDeDyIdx];
    synTensorDescriptor wDesc    = m_tensorDescs[opReluOutIdx];
    synTensorDescriptor dedxDesc = m_tensorDescs[dedxOutIdx];

    auto dedyData = m_hostBuffers[opDeDyIdx];
    auto dedxData = m_hostBuffers[dedxOutIdx];

    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDX(dedyDesc,
                         (char*)dedyData,
                         wDesc,
                         (char*)relu1Output.data(), /* relu output is W */
                         dedxDesc,
                         (char*)dedxData,
                         params,
                         wrongIdx,
                         m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[dedxOutIdx].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        m_tensorDescs[dedxOutIdx].m_dataType,
                                                        m_hostBuffers[dedxOutIdx]);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, dedx_1x1_L2)
{
    const unsigned b = 10;
    const unsigned h = 64, w = 64, c = 130, k = 130;
    const unsigned r = 1, s = 1;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned wSizes[]       = {k, c, s ,r};
    unsigned dedxOutSizes[] = {c, w, h, b};

    unsigned expBatchSlice = 1;
    unsigned expSpatialSlice = 64;
    std::string originalSRAMCap;
    originalSRAMCap.reserve(32);
    std::stringstream slicerCapVal;
    slicerCapVal << (2 * expBatchSlice * (h * w * expSpatialSlice + h * w * expSpatialSlice) + (k * c))* sizeof(float) ;
    synConfigurationGet("SRAM_SLICER_MAX_CAPACITY_BYTES", &originalSRAMCap[0], 32);
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    unsigned opDeDyIdx  = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes,    4, syn_type_float, nullptr, "dedy");
    unsigned opWIdx     = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes,       4, syn_type_float, nullptr, "w");
    unsigned dedxOutIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, dedxOutSizes, 4, syn_type_float, nullptr, "dedx");

    synConvolutionParams params;

    addNodeToGraph(NodeFactory::deDxNodeTypeName, {opDeDyIdx, opWIdx}, {dedxOutIdx}, &params, sizeof(params), "dedx");

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[opDeDyIdx];
    synTensorDescriptor xDesc    = m_tensorDescs[opWIdx];
    synTensorDescriptor dedxDesc = m_tensorDescs[dedxOutIdx];

    auto xData    = m_hostBuffers[opWIdx];
    auto dedyData = m_hostBuffers[opDeDyIdx];
    auto dedxData = m_hostBuffers[dedxOutIdx];

    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDX(dedyDesc,
                         (char*)dedyData,
                         xDesc,
                         (char*)xData,
                         dedxDesc,
                         (char*)dedxData,
                         params,
                         wrongIdx,
                         m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[dedxOutIdx].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        m_tensorDescs[dedxOutIdx].m_dataType,
                                                        m_hostBuffers[dedxOutIdx]);
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", originalSRAMCap.c_str());
}

TEST_F_GC(SynGaudiSRAMSlicingTest, relu_dedx_1x1_relu_ASIC)
{
    const unsigned b = 10;
    const unsigned h = 64, w = 64, c = 130, k = 130;
    const unsigned r = 1, s = 1;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned wSizes[]       = {k, c, s ,r};
    unsigned dedxOutSizes[] = {c, w, h, b};

    synConvolutionParams params;
    unsigned expBatchSlice = 1;
    unsigned expSpatialSlice = 64;
    std::string originalSRAMCap;
    originalSRAMCap.reserve(32);
    std::stringstream slicerCapVal;
    slicerCapVal << (2 * expBatchSlice * (h * w * expSpatialSlice + h * w * expSpatialSlice) + (k * c))* sizeof(float) ;
    synConfigurationGet("SRAM_SLICER_MAX_CAPACITY_BYTES", &originalSRAMCap[0], 32);
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    unsigned opReluInIdx  = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes, 4, syn_type_float, nullptr, "relu1_in");
    unsigned opReluOutIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, wSizes, 4, syn_type_float, nullptr, "relu1_out");

    unsigned opWIdx       = connectOutputTensorToInputTensor(opReluOutIdx);
    unsigned opDeDyIdx    = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes,    4, syn_type_float, nullptr, "dedy");
    unsigned dedxOutIdx   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, dedxOutSizes, 4, syn_type_float, nullptr, "dedx");
    unsigned relu2InIdx   = connectOutputTensorToInputTensor(dedxOutIdx);
    unsigned relu2OutIdx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, dedxOutSizes, 4, syn_type_float, nullptr, "relu2Out");


    addNodeToGraph("relu_fwd_f32",                {opReluInIdx},       {opReluOutIdx}, nullptr, 0,              "relu_1");
    addNodeToGraph(NodeFactory::deDxNodeTypeName, {opDeDyIdx, opWIdx}, {dedxOutIdx},   &params, sizeof(params), "dedx");
    addNodeToGraph("relu_fwd_f32",                {relu2InIdx},        {relu2OutIdx},  nullptr, 0,              "relu_2");

    compileAndRun();

    unsigned           relu1OutputSize = k*c*s*r;
    float*             relu1Input      = (float *)m_hostBuffers[opReluInIdx];
    std::vector<float> relu1Output(relu1OutputSize);

    for (int i = 0; i < k*c*s*r ; i++)
    {
        relu1Output[i] = ((relu1Input[i] > 0) ? relu1Input[i] : 0);
    }


    synTensorDescriptor dedyDesc = m_tensorDescs[opDeDyIdx];
    synTensorDescriptor wDesc    = m_tensorDescs[opReluOutIdx];
    synTensorDescriptor dedxDesc = m_tensorDescs[dedxOutIdx];

    auto dedyData = m_hostBuffers[opDeDyIdx];
    auto dedxData = m_hostBuffers[dedxOutIdx];

    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDX(dedyDesc,
                         (char*)dedyData,
                         wDesc,
                         (char*)relu1Output.data(), /* relu output is W */
                         dedxDesc,
                         (char*)dedxData,
                         params,
                         wrongIdx,
                         m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedxDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedxDesc.m_dataType,
                                                        m_hostBuffers[dedxOutIdx]);

    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", originalSRAMCap.c_str());
}

TEST_F_GC(SynGaudiSRAMSlicingTest, relu_dedx_filter_relu_L2)
{
    const unsigned b = 10;
    const unsigned r = 3, s = 3;
    const unsigned h = 64, w = 64, c = 130, k = 130;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned wSizes[]       = {k, c, s ,r};
    unsigned dedxOutSizes[] = {c, w + s - 1, h + r - 1 ,b};

    unsigned expBatchSlice = 1;
    unsigned expSpatialSlice = 64;
    std::string originalSRAMCap;
    originalSRAMCap.reserve(32);
    std::stringstream slicerCapVal;
    slicerCapVal << (2 * expBatchSlice * (h * w * expSpatialSlice + h * w * expSpatialSlice) + (k * c))* sizeof(float) ;
    synConfigurationGet("SRAM_SLICER_MAX_CAPACITY_BYTES", &originalSRAMCap[0], 32);
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    synConvolutionParams params;
    params.kH = r;
    params.kW = s;

    unsigned opReluInIdx  = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes, 4, syn_type_float, nullptr, "relu1_in");
    unsigned opReluOutIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, wSizes, 4, syn_type_float, nullptr, "relu1_out");
    unsigned opWIdx       = connectOutputTensorToInputTensor(opReluOutIdx);
    unsigned opDeDyIdx    = createPersistTensor(INPUT_TENSOR,  MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes,    4, syn_type_float, nullptr, "dedy");
    unsigned dedxOutIdx   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, dedxOutSizes, 4, syn_type_float, nullptr, "dedx");
    unsigned relu2InIdx   = connectOutputTensorToInputTensor(dedxOutIdx);
    unsigned relu2OutIdx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,             nullptr, dedxOutSizes, 4, syn_type_float, nullptr, "relu2Out");

    addNodeToGraph("relu_fwd_f32",                {opReluInIdx},       {opReluOutIdx}, nullptr, 0,              "relu_1");
    addNodeToGraph(NodeFactory::deDxNodeTypeName, {opDeDyIdx, opWIdx}, {dedxOutIdx},   &params, sizeof(params), "dedx");
    addNodeToGraph("relu_fwd_f32",                {relu2InIdx},        {relu2OutIdx},  nullptr, 0,              "relu_2");

    compileAndRun();

    unsigned           relu1OutputSize = k*c*s*r;
    float*             relu1Input      = (float *)m_hostBuffers[opReluInIdx];
    std::vector<float> relu1Output(relu1OutputSize);

    for (int i = 0; i < relu1OutputSize ; i++)
    {
        relu1Output[i] = ((relu1Input[i] > 0) ? relu1Input[i] : 0);
    }


    synTensorDescriptor dedyDesc = m_tensorDescs[opDeDyIdx];
    synTensorDescriptor wDesc    = m_tensorDescs[opReluOutIdx];
    synTensorDescriptor dedxDesc = m_tensorDescs[dedxOutIdx];

    auto dedyData = m_hostBuffers[opDeDyIdx];
    auto dedxData = m_hostBuffers[dedxOutIdx];

    unsigned dedxOutSize = std::accumulate(dedxOutSizes, dedxOutSizes + 4, 1, std::multiplies<unsigned>());
    float*   cpuResults  = new float[dedxOutSize];
    calculateDEDX(dedyDesc,
                  (char*)dedyData,
                  wDesc,
                  (char*)relu1Output.data(),
                  dedxDesc,
                  (char*)cpuResults,
                  params,
                  m_deviceType);
    validateResults(dedxDesc, (char*)dedxData, (char*)cpuResults);
    delete[] cpuResults;

    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", originalSRAMCap.c_str());
}

TEST_F_GC(SynGaudiSRAMSlicingTest, dedx_stitched_to_dedw_bundle_L2)
{
    const unsigned
        b = 1,
        h = 1,
        w = 2048,
        k = 256,
        c = 128,
        r = 1,
        s = 1;

    unsigned dySizes[]  = {k, w, h, b};
    unsigned xSizes[]   = {c, w, h, b};
    unsigned wghSizes[] = {k, c, s, r};

    unsigned dy  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dySizes, 4, syn_type_float, nullptr, "dedy");
    unsigned x   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "x");
    unsigned wgh = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wghSizes, 4, syn_type_float, nullptr, "wgh");
    unsigned dw  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wghSizes, 4, syn_type_float, nullptr, "dedw");
    unsigned dx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes, 4, syn_type_float, nullptr, "dedx");

    synConvolutionParams params;
    addNodeToGraph(NodeFactory::deDxNodeTypeName, {dy, wgh}, {dx}, &params, sizeof(params), "dedx");
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dy, x}, {dw}, &params, sizeof(params), "dedw");

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[dy];
    synTensorDescriptor xDesc = m_tensorDescs[x];
    synTensorDescriptor wDesc = m_tensorDescs[wgh];
    synTensorDescriptor dedwDesc = m_tensorDescs[dw];
    synTensorDescriptor dedxDesc = m_tensorDescs[dx];

    auto xData = m_hostBuffers[x];
    auto wData = m_hostBuffers[wgh];
    auto dedyData = m_hostBuffers[dy];
    auto dedwData = m_hostBuffers[dw];
    auto dedxData = m_hostBuffers[dx];

    CoordArray wrongIdx = {0};
    bool       retDX    = checkDEDX(dedyDesc,
                           (char*)dedyData,
                           wDesc,
                           (char*)wData,
                           dedxDesc,
                           (char*)dedxData,
                           params,
                           wrongIdx,
                           m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, m_tensorDescs[dx].m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(retDX) << "DEDX: Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                    << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        m_tensorDescs[dx].m_dataType,
                                                        m_hostBuffers[dx]);
    wrongIdx = {0};
    bool retDW = checkDEDW(dedyDesc,
                           (char*)dedyData,
                           xDesc,
                           (char*)xData,
                           dedwDesc,
                           (char*)dedwData,
                           params,
                           wrongIdx,
                           m_deviceType);
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(retDW) << "DEDW: Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                    << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedwDesc.m_dataType,
                                                    dedwData);

}

// This test fails on gaudi3 because of the difference in the order of summation between the cpu
// calculator and the chip when running with cd concurrency in the mme.
// The test passes with ENABLE_MME_CD_CONCURRENCY=false.
TEST_F_GC(SynGaudiSRAMSlicingTest, dedw_sliced_on_all_dims_L2, {synDeviceGaudi, synDeviceGaudi2})
{
    unsigned b = 10;
    unsigned h = 64, w = 64, c = 130, k = 130;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned xSizes[]       = {c, w, h ,b};
    unsigned dedwOutSizes[] = {k, c, 1, 1};

    unsigned expBatchSlice = 1;
    unsigned expSpatialSlice = 64;
    std::stringstream slicerCapVal;
    slicerCapVal << (2 * expBatchSlice * (h * w * expSpatialSlice + h * w * expSpatialSlice) + (k * c))* sizeof(float) ;
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    unsigned opDeDyIdx  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes, 4, syn_type_float, nullptr, "dedy");
    unsigned opXIdx     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "x");
    unsigned dedwOutIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedwOutSizes, 4, syn_type_float, nullptr, "dedw");
    synConvolutionParams params;
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {opDeDyIdx, opXIdx}, {dedwOutIdx}, &params, sizeof(params), "dedw");

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[opDeDyIdx];
    synTensorDescriptor xDesc = m_tensorDescs[opXIdx];
    synTensorDescriptor dedwDesc = m_tensorDescs[dedwOutIdx];

    auto xData = m_hostBuffers[opXIdx];
    auto dedyData = m_hostBuffers[opDeDyIdx];
    auto dedwData = m_hostBuffers[dedwOutIdx];
    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDW(dedyDesc,
                         (char*)dedyData,
                         xDesc,
                         (char*)xData,
                         dedwDesc,
                         (char*)dedwData,
                         params,
                         wrongIdx,
                         m_deviceType);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedwDesc.m_dataType,
                                                        dedwData);

}

TEST_F_GC(SynGaudiSRAMSlicingTest, dedw_sliced_on_all_dims2_L2)
{
    unsigned b = 16;
    unsigned h1 = 7, w1 = 7, h2 = 14, w2 = 14, c = 1024 , k = 2048;
    unsigned dedySizes[]    = {k, w1, h1, b};
    unsigned xSizes[]       = {c, w2, h2 ,b};
    unsigned dedwOutSizes[] = {k, c, 1, 1};

    unsigned opDeDyIdx  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes, 4, syn_type_float, nullptr, "dedy");
    unsigned opXIdx     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "x");
    unsigned dedwOutIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedwOutSizes, 4, syn_type_float, nullptr, "dedw");
    synConvolutionParams params;
    params.dW = 2; params.dH = 2;
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {opDeDyIdx, opXIdx}, {dedwOutIdx}, &params, sizeof(params), "dedw");

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[opDeDyIdx];
    synTensorDescriptor xDesc = m_tensorDescs[opXIdx];
    synTensorDescriptor dedwDesc = m_tensorDescs[dedwOutIdx];

    auto xData = m_hostBuffers[opXIdx];
    auto dedyData = m_hostBuffers[opDeDyIdx];
    auto dedwData = m_hostBuffers[dedwOutIdx];
    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDW(dedyDesc,
                         (char*)dedyData,
                         xDesc,
                         (char*)xData,
                         dedwDesc,
                         (char*)dedwData,
                         params,
                         wrongIdx,
                         m_deviceType);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedwDesc.m_dataType,
                                                        dedwData);
}

// This test fails on gaudi3 because of the difference in the order of summation between the cpu
// calculator and the chip when running with cd concurrency in the mme.
// The test passes with ENABLE_MME_CD_CONCURRENCY=false.
TEST_F_GC(SynGaudiSRAMSlicingTest, dedw_sliced_on_all_dims_with_tpc_consumer_L2, {synDeviceGaudi, synDeviceGaudi2})
{
    unsigned b = 10;
    unsigned h = 64, w = 64, c = 130, k = 130;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned xSizes[]       = {c, w, h ,b};
    unsigned dedwOutSizes[] = {k, c, 1, 1};
    unsigned addInSizes[] = {k, c, 1, 1};
    unsigned opOutSizes[] = {k, c, 1, 1};

    unsigned expBatchSlice = 1;
    unsigned expSpatialSlice = 64;
    std::stringstream slicerCapVal;
    slicerCapVal << (2 * expBatchSlice * (h * w * expSpatialSlice + h * w * expSpatialSlice) + (k * c))* sizeof(float) ;
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    unsigned opDeDyIdx  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes, 4, syn_type_float, nullptr, "dedy");
    unsigned opXIdx     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "x");
    unsigned dedwOutIdx = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedwOutSizes, 4, syn_type_float);
    unsigned dedwInIdx = connectOutputTensorToInputTensor(dedwOutIdx);
    unsigned addInIdx   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, addInSizes, 4, syn_type_float, nullptr, "addIn");
    unsigned opOutIdx   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, syn_type_float, nullptr, "addOut");
    synConvolutionParams params;
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {opDeDyIdx, opXIdx}, {dedwOutIdx}, &params, sizeof(params), "dedw");
    addNodeToGraph("add_fwd_f32", {dedwInIdx, addInIdx}, {opOutIdx}, nullptr, 0, "add");

    compileAndRun();

    const auto dedyDesc = static_cast<synTensorDescriptor>(m_tensorDescs[opDeDyIdx]);
    const auto xDesc    = static_cast<synTensorDescriptor>(m_tensorDescs[opXIdx]);
    const auto dedwDesc = static_cast<synTensorDescriptor>(m_tensorDescs[dedwOutIdx]);

    auto xData = m_hostBuffers[opXIdx];
    auto dedyData = m_hostBuffers[opDeDyIdx];

    // perform subtraction of the added tensor to produce the dedw output.
    auto opOutData = m_hostBuffers[opOutIdx];
    uint64_t dedwOutSize = getMemorySize(dedwDesc.m_sizes, dedwDesc.m_dataType, dedwDesc.m_dims) / sizeof(float);
    std::vector<float> calcDeDwOut(dedwOutSize);
    for (uint64_t i = 0; i < dedwOutSize; ++i)
    {
        float val = *((float*)opOutData + i);
        float addedVal = *((float*) m_hostBuffers[addInIdx] + i);
        calcDeDwOut[i] = val - addedVal;
    }
    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDW(dedyDesc,
                         (char*)dedyData,
                         xDesc,
                         (char*)xData,
                         dedwDesc,
                         (char*)calcDeDwOut.data(),
                         params,
                         wrongIdx,
                         m_deviceType);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedwDesc.m_dataType,
                                                        calcDeDwOut.data());
}

TEST_F_GC(SynGaudiSRAMSlicingTest,
          dedw_sliced_on_all_dims_with_tpc_consumer_and_tpc_producer_L2,
          {synDeviceGaudi, synDeviceGaudi2})
{
    unsigned b = 10;
    unsigned h = 64, w = 64, c = 130, k = 130;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned xSizes[]       = {c, w, h ,b};
    unsigned dedwOutSizes[] = {k, c, 1, 1};
    unsigned addInSizes[] = {k, c, 1, 1};
    unsigned opOutSizes[] = {k, c, 1, 1};

    unsigned expBatchSlice = 1;
    unsigned expSpatialSlice = 64;
    std::stringstream slicerCapVal;
    // expected - slice on each batch, and slice 64 chunks from the spatial dims, double-buffer on the inputs.
    slicerCapVal << (2 * expBatchSlice * (h * w * expSpatialSlice + h * w * expSpatialSlice) + (k * c))* sizeof(float) ;
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    unsigned opDeDyIdx     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes, 4, syn_type_float, nullptr, "dedy");
    unsigned reluOutIdx    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedySizes, 4, syn_type_float);
    unsigned reluOutInIdx  = connectOutputTensorToInputTensor(reluOutIdx);
    unsigned opXIdx        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "x");
    unsigned dedwOutIdx    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedwOutSizes, 4, syn_type_float);
    unsigned dedwInIdx     = connectOutputTensorToInputTensor(dedwOutIdx);
    unsigned addInIdx      = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, addInSizes, 4, syn_type_float, nullptr, "addIn");
    unsigned opOutIdx      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, syn_type_float, nullptr, "addOut");
    synConvolutionParams params;
    addNodeToGraph("relu_fwd_f32", {opDeDyIdx}, {reluOutIdx});
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {reluOutInIdx, opXIdx}, {dedwOutIdx}, &params, sizeof(params), "dedw");
    addNodeToGraph("add_fwd_f32", {dedwInIdx, addInIdx}, {opOutIdx}, nullptr, 0, "add");

    compileAndRun();

    const auto dedyDesc = static_cast<synTensorDescriptor>(m_tensorDescs[opDeDyIdx]);
    const auto xDesc    = static_cast<synTensorDescriptor>(m_tensorDescs[opXIdx]);
    const auto dedwDesc = static_cast<synTensorDescriptor>(m_tensorDescs[dedwOutIdx]);

    auto xData = m_hostBuffers[opXIdx];
    auto dedyData = m_hostBuffers[opDeDyIdx];

    // perform manual calculation of relu on dedy input
    uint64_t reluOutSize = getMemorySize(dedyDesc.m_sizes, dedyDesc.m_dataType, dedyDesc.m_dims) / sizeof(float);
    uint64_t dedwOutSize = getMemorySize(dedwDesc.m_sizes, dedwDesc.m_dataType, dedwDesc.m_dims) / sizeof(float);
    std::vector<float> calcReluOut(reluOutSize), calcReluOut2(reluOutSize), calcDeDwOut(dedwOutSize);
    calculateRelu(dedyDesc, dedyData, dedyDesc, calcReluOut.data());

    // perform subtraction of the added tensor to produce the dedw output.
    auto opOutData = m_hostBuffers[opOutIdx];
    for (uint64_t i = 0; i < dedwOutSize; ++i)
    {
        float val = *((float*)opOutData + i);
        float addedVal = *((float*) m_hostBuffers[addInIdx] + i);
        calcDeDwOut[i] = val - addedVal;
    }
    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDW(dedyDesc,
                         (char*)calcReluOut.data(),
                         xDesc,
                         (char*)xData,
                         dedwDesc,
                         (char*)calcDeDwOut.data(),
                         params,
                         wrongIdx,
                         m_deviceType);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedwDesc.m_dataType,
                                                        calcDeDwOut.data());
}

TEST_F_GC(SynGaudiSRAMSlicingTest, sram_slicing_for_relu_conv_trivial_fp32_ASIC)
{
    testReluConvSlicing("18000000");
}

TEST_F_GC(SynGaudiSRAMSlicingTest, sram_slicing_for_relu_conv_fp32_L2)
{
    testReluConvSlicing("10000000");
}

TEST_F_GC(SynGaudiSRAMSlicingTest,
          sram_slicing_for_relu_conv_trivial_fp32_check_intermediate_L2,
          {synDeviceGaudi, synDeviceGaudi2})
{
    testReluConvSlicing("18000000", true);
}

TEST_F_GC(SynGaudiSRAMSlicingTest,
          sram_slicing_for_relu_conv_fp32_check_intermediate_ASIC,
          {synDeviceGaudi, synDeviceGaudi2})
{
    testReluConvSlicing("10000000", true);
}

TEST_F_GC(SynGaudiSRAMSlicingTest,
          sram_slicing_for_relu_conv_trivial_with_eviction_fp32_ASIC,
          {synDeviceGaudi, synDeviceGaudi2})
{
    testReluConvWithEvictionSlicing("18000000");
}

TEST_F_GC(SynGaudiSRAMSlicingTest,
          sram_slicing_for_relu_conv_with_eviction_fp32_ASIC_,
          {synDeviceGaudi, synDeviceGaudi2})
{
    testReluConvWithEvictionSlicing("10000000");
}

TEST_F_GC(SynGaudiSRAMSlicingConsistency,
          sram_slicing_for_relu_conv_with_eviction_fp32_compilation_consistency_ASIC,
          {synDeviceGaudi, synDeviceGaudi2})
{
    testReluConvWithEvictionSlicingConsistency("10000000");
}

// DEDW node. output is defined as bf16, but since the common dim solver is chosen, the results are calculated using
// "partials" aka reduction. The test passes because we force the output of the reduction to be fp32, otherwise the
// results were incorrect. Since the user defined the original tensor ot be bf16, a cast is added from fp32 to bf16
// as the last node.
TEST_F_GC(SynGaudiSRAMSlicingTest, common_dim_solver_original_output_is_bf16)
{
    // since b*h*w=10*64*64 is bigger than SlicingBrain::knobs.minCDSizeForPartials = 512, the common dim solver is used
    const unsigned b = 10, h = 64, w = 64;
    const unsigned c = 50, k = 2;
    unsigned dedySizes[]    = {k, w, h, b};
    unsigned xSizes[]       = {c, w, h ,b};
    unsigned dedwOutSizes[] = {k, c, 1, 1};

    unsigned dedyTensor    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes, 4,
                                                 syn_type_bf16, nullptr, "dedy");
    unsigned xTensor       = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4,
                                                 syn_type_bf16, nullptr, "x");
    unsigned dedwOutTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedwOutSizes, 4,
                                                 syn_type_bf16, nullptr, "dedw_out");
    synConvolutionParams params;
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dedyTensor, xTensor}, {dedwOutTensor}, &params, sizeof(params),
                  "dedw");

    compileAndRun();

    synTensorDescriptor dedyDesc = m_tensorDescs[dedyTensor];
    synTensorDescriptor xDesc = m_tensorDescs[xTensor];
    synTensorDescriptor dedwDesc = m_tensorDescs[dedwOutTensor];
    char* dedyData = static_cast<char*>(m_hostBuffers[dedyTensor]);
    char* xData = static_cast<char*>(m_hostBuffers[xTensor]);
    char* dedwOutData = static_cast<char*>(m_hostBuffers[dedwOutTensor]);

    CoordArray wrongIdx = {0};
    // In Gaudi2 the test does not perform slicing, and therefore element-wise checking fails (because cd concurrency
    // is applied when the test is in bf16 and the dedw is not sliced). SW-137589
    // Therefore we need to apply a Pearson tensor-global checking instead of element-wise checking
    bool       ret = checkDEDW(dedyDesc, dedyData, xDesc, xData, dedwDesc, dedwOutData, params, wrongIdx, m_deviceType, true);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, dedwDesc.m_dataType, dedwOutData);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, big_conv_relu_x5_batch_1_ASIC)
{
    testBigConvReluX5(1);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, DISABLED_big_conv_relu_x5_batch_8_ASIC, {synDeviceGaudi, synDeviceGaudi2})
{
    testBigConvReluX5(8);
}

TEST_F_GC(SynGaudiSRAMSlicingTest,
          tpc_producer_stitched_to_wide_when_dedw_sliced_on_multiple_dims_L2,
          {synDeviceGaudi, synDeviceGaudi2})
{
    const unsigned b = 3, hX = 140, wX = 140;
    const unsigned c = 150, k = 200;
    synConvolutionParams params;
    params.dW = 2, params.dH = 2;

    unsigned xSizes[] = {c, wX, hX, b};
    unsigned dedwSizes[] = {k, c, params.kW, params.kH};
    unsigned dedySizes[] = {k,
                                  convOutputDimSize(wX, params.kW, params.dW, params.padL + params.padR, params.dilW),
                                  convOutputDimSize(hX, params.kH, params.dH, params.padT + params.padB, params.dilH),
                                  b}; // {k,wX/2,hX/2,b}

    std::stringstream maxSramCap;
    maxSramCap << 18 * 1024 * 1024;
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", maxSramCap.str().c_str());

    unsigned reluIn     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes, 4, syn_type_float, nullptr, "reluIn");
    unsigned reluOut    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedySizes, 4, syn_type_float);
    unsigned dedy       = connectOutputTensorToInputTensor(reluOut);
    unsigned x          = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "x");
    unsigned dedwOut    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedwSizes, 4, syn_type_float, nullptr, "dedwOut");

    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "relu");
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dedy, x}, {dedwOut}, &params, sizeof(params), "dedw");

    compileAndRun();

    const auto reluDesc = static_cast<synTensorDescriptor>(m_tensorDescs[reluIn]);
    const auto xDesc    = static_cast<synTensorDescriptor>(m_tensorDescs[x]);
    const auto dedwDesc = static_cast<synTensorDescriptor>(m_tensorDescs[dedwOut]);

    auto xData = m_hostBuffers[x];
    auto reluInData = m_hostBuffers[reluIn];

    // perform manual calculation of rel
    uint64_t reluOutSize = getMemorySize(reluDesc.m_sizes, reluDesc.m_dataType, reluDesc.m_dims) / sizeof(float);
    uint64_t dedwOutSize = getMemorySize(dedwDesc.m_sizes, dedwDesc.m_dataType, dedwDesc.m_dims) / sizeof(float);
    std::vector<float> calcReluOut(reluOutSize), calcDeDwOut(dedwOutSize);
    calculateRelu(reluDesc, reluInData, reluDesc, calcReluOut.data());

    auto opOutData = m_hostBuffers[dedwOut];
    for (uint64_t i = 0; i < dedwOutSize; ++i)
    {
        float val = *((float*)opOutData + i);
        calcDeDwOut[i] = val;
    }
    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDW(reluDesc,
                         (char*)calcReluOut.data(),
                         xDesc,
                         (char*)xData,
                         dedwDesc,
                         (char*)calcDeDwOut.data(),
                         params,
                         wrongIdx,
                         m_deviceType);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedwDesc.m_dataType,
                                                        calcDeDwOut.data());
}

TEST_F_GC(SynGaudiSRAMSlicingTest,
          tpc_producer_stitched_to_narrow_when_dedw_sliced_on_multiple_dims_L2,
          {synDeviceGaudi, synDeviceGaudi2})
{
    const unsigned b = 3, hX = 140, wX = 140;
    const unsigned c = 150, k = 200;
    synConvolutionParams params;
    params.dW = 2, params.dH = 2;

    unsigned xSizes[] = {c, wX, hX, b};
    unsigned dedwSizes[] = {k, c, params.kW, params.kH};
    unsigned dedySizes[] = {k,
                            convOutputDimSize(wX, params.kW, params.dW, params.padL + params.padR, params.dilW),
                            convOutputDimSize(hX, params.kH, params.dH, params.padT + params.padB, params.dilH),
                            b}; // {k,wX/2,hX/2,b}

    std::stringstream maxSramCap;
    maxSramCap << 18 * 1024 * 1024;
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", maxSramCap.str().c_str());

    unsigned reluIn     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_float, nullptr, "reluIn");
    unsigned reluOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes, 4, syn_type_float);
    unsigned dedy       = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dedySizes, 4, syn_type_float, nullptr, "dedy");
    unsigned x          = connectOutputTensorToInputTensor(reluOut);
    unsigned dedwOut    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dedwSizes, 4, syn_type_float, nullptr, "dedwOut");

    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "relu");
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dedy, x}, {dedwOut}, &params, sizeof(params), "dedw");

    compileAndRun();

    const auto reluDesc = static_cast<synTensorDescriptor>(m_tensorDescs[reluIn]);
    const auto dedyDesc = static_cast<synTensorDescriptor>(m_tensorDescs[dedy]);
    const auto dedwDesc = static_cast<synTensorDescriptor>(m_tensorDescs[dedwOut]);

    auto dedyData = m_hostBuffers[dedy];
    auto reluInData = m_hostBuffers[reluIn];

    // perform manual calculation of relu
    uint64_t reluOutSize = getMemorySize(reluDesc.m_sizes, reluDesc.m_dataType, reluDesc.m_dims) / sizeof(float);
    uint64_t dedwOutSize = getMemorySize(dedwDesc.m_sizes, dedwDesc.m_dataType, dedwDesc.m_dims) / sizeof(float);
    std::vector<float> calcReluOut(reluOutSize), calcDeDwOut(dedwOutSize);
    calculateRelu(reluDesc, reluInData, reluDesc, calcReluOut.data());

    auto opOutData = m_hostBuffers[dedwOut];
    for (uint64_t i = 0; i < dedwOutSize; ++i)
    {
        float val = *((float*)opOutData + i);
        calcDeDwOut[i] = val;
    }
    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDW(dedyDesc,
                         (char*)dedyData,
                         reluDesc,
                         (char*)calcReluOut.data(),
                         dedwDesc,
                         (char*)calcDeDwOut.data(),
                         params,
                         wrongIdx,
                         m_deviceType);
    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dedwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx,
                                                        dedwDesc.m_dataType,
                                                        calcDeDwOut.data());
}

TEST_F_GC(SynGaudiSRAMSlicingTest, mme_slave_producer_consumer_L2)
{
    GlobalConfTestSetter prodExpSetter("SRAM_SLICER_SHARED_MME_INPUT_PRODUCER_EXPANSION_ENABLED", "true");
    GlobalConfTestSetter consExpSetter("SRAM_SLICER_SHARED_MME_INPUT_CONSUMER_EXPANSION_ENABLED", "true");

    const unsigned b1 = 1, h1 = 1, w1 = 6272, c1 = 512;
    const unsigned r = 1, s = 1, k = 256;

    synConvolutionParams convParams;
    convParams.kH = r;
    convParams.kW = s;

    unsigned wOut = convOutputDimSize(w1, convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW);
    unsigned hOut = convOutputDimSize(h1, convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH);

    unsigned opASizes[]   = {c1, w1, h1, b1};
    unsigned opBSizes[]   = {k, c1, s, r};
    unsigned opOutSizes[] = {k, wOut, hOut, b1};


    unsigned opAIdx    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, opASizes, 4, syn_type_float, nullptr, "sharedInput");
    unsigned opBIdx    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, opBSizes, 4, syn_type_float, nullptr, "opB");
    unsigned opOutIdx = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, syn_type_float);

    unsigned reluConsumer1In = connectOutputTensorToInputTensor(opOutIdx);
    unsigned reluConsumer1Out  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, syn_type_float, nullptr, "reluConsumer1Out");

    unsigned relu2In   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, opBSizes, 4, syn_type_float, nullptr,  "relu2In");
    unsigned relu2Out  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opBSizes, 4, syn_type_float);
    unsigned opB2Idx   = connectOutputTensorToInputTensor(relu2Out);
    unsigned opOut2Idx = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, syn_type_float);

    unsigned relu2ConsumerIn  = connectOutputTensorToInputTensor(opOut2Idx);
    unsigned relu2ConsumerOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, opOutSizes, 4, syn_type_float, nullptr, "reluConsumer2Out");

    synConvolutionParams params;
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opAIdx, opBIdx},   {opOutIdx},  &params, sizeof(params), "conv1");
    addNodeToGraph("relu_fwd_f32",                       {reluConsumer1In},  {reluConsumer1Out},  nullptr, 0,      "relu_1_consumer");

    addNodeToGraph("relu_fwd_f32",                       {relu2In},          {relu2Out},  nullptr, 0,              "relu_2_producer");
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {opAIdx, opB2Idx}, {opOut2Idx}, &params, sizeof(params), "conv2");
    addNodeToGraph("relu_fwd_f32",                       {relu2ConsumerIn},  {relu2ConsumerOut},  nullptr, 0,      "relu_2_consumer");

    compileAndRun();

    //Check results - conv 1
    const synTensorDescriptor&  conv1OpADescriptor = m_tensorDescs[opAIdx];
    const synTensorDescriptor&  conv1OpBDescriptor = m_tensorDescs[opBIdx];
    const synTensorDescriptor&  conv1OpOut         = m_tensorDescs[reluConsumer1Out];

    auto opAData   =  m_hostBuffers[opAIdx];
    auto opBData   =  m_hostBuffers[opBIdx];
    auto opOutData =  m_hostBuffers[reluConsumer1Out];


    CoordArray wrongIdx = {0};
    bool       ret      = checkFwdConvolution(conv1OpADescriptor,
                                   (char*)opAData,
                                   conv1OpBDescriptor,
                                   (char*)opBData,
                                   conv1OpOut,
                                   (char*)opOutData,
                                   params,
                                   wrongIdx,
                                   m_deviceType);
    ASSERT_TRUE(ret);

    //Check results - conv 2
    const synTensorDescriptor&  conv2OpADescriptor = m_tensorDescs[opAIdx];
    const synTensorDescriptor&  conv2OpBDescriptor = m_tensorDescs[relu2In];
    const synTensorDescriptor&  conv2OpOut         = m_tensorDescs[relu2ConsumerOut];

    auto conv2OpAData   =  m_hostBuffers[opAIdx];
    auto conv2OpBData   =  m_hostBuffers[relu2In];
    auto conv2OpOutData =  m_hostBuffers[relu2ConsumerOut];

    ret = checkFwdConvolution(conv2OpADescriptor,
                              (char*)conv2OpAData,
                              conv2OpBDescriptor,
                              (char*)conv2OpBData,
                              conv2OpOut,
                              (char*)conv2OpOutData,
                              params,
                              wrongIdx,
                              m_deviceType);
    ASSERT_TRUE(ret);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, sram_slicing_for_conv_dedw_L2)
{
    // create sharedInput tensor
    unsigned sharedInput_sizes[] = {256,64,64,2};
    unsigned sharedInput = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "sharedInput",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        sharedInput_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create convInput tensor
    unsigned convInput_sizes[] = {256,256,3,3};
    unsigned convInput = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "convInput",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        convInput_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create convOutput tensor
    unsigned convOutput_sizes[] = {256,64,64,2};
    unsigned convOutput = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "convOutput",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        convOutput_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char syn_3413_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,105,117,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,189,202,200,53,127,0,0,0,0,0,0,254,127,0,0};
    addNodeToGraph("spatial_convolution", {sharedInput, convInput}, {convOutput}, (void*)syn_3413_params, 88, "syn_3413");

    // create dedwInput tensor
    unsigned dedwInput_sizes[] = {256,64,64,2};
    unsigned dedwInput = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "dedwInput",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        dedwInput_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create dedwOutput tensor
    unsigned dedwOutput_sizes[] = {256,256,3,3};
    unsigned dedwOutput = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "dedwOutput",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        dedwOutput_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char syn_3879_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,189,202,200,53,127,0,0,0,0,0,0,254,127,0,0};
    addNodeToGraph("dedw", {dedwInput, sharedInput}, {dedwOutput}, (void*)syn_3879_params, 88, "syn_3879");


    compileTopology();
}

TEST_F_GC(SynGaudiSRAMSlicingTest, dedw_w_broadcast_producer_L2)
{
    unsigned dySize[] = {1152, 8192, 1, 1};
    unsigned  xSize[] = {192, 8192, 1, 1};
    unsigned dwSize[] = {1152, 192, 1, 1};

    unsigned bxSize[] = {1, 1, 1, 1};

    auto addIn0 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      dySize,
                                      ARRAY_SIZE(dySize),
                                      syn_type_float,
                                      nullptr,
                                      "ADD_IN");
    auto addIn1 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      bxSize,
                                      ARRAY_SIZE(bxSize),
                                      syn_type_float,
                                      nullptr,
                                      "BXTensor");

    auto dy = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, dySize, ARRAY_SIZE(dySize), syn_type_float);

    addNodeToGraph("add_fwd_f32", {addIn0, addIn1}, {dy}, nullptr, 0, "Add");

    auto x = createPersistTensor(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 xSize,
                                 ARRAY_SIZE(xSize),
                                 syn_type_float,
                                 nullptr,
                                 "X");

    auto dw = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_NONE,
                                  nullptr,
                                  dwSize,
                                  ARRAY_SIZE(dwSize),
                                  syn_type_float,
                                  nullptr,
                                  "DW");

    synConvolutionParams dedwParams {};
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dy, x}, {dw}, &dedwParams, sizeof dedwParams, "DEDW");

    compileAndRun();
    ASSERT_FALSE(HasFailure());

    auto bxElement = *castHostBuffer<float>(addIn1);
    auto* dyRef = castHostBuffer<float>(addIn0);
    size_t dyElements = multiplyElements(dySize, dySize + ARRAY_SIZE(dySize));
    for (size_t idx = 0; idx < dyElements; idx++)
    {
        dyRef[idx] += bxElement;
    }
    CoordArray outIdx{};
    bool       check = checkDEDW(m_tensorDescs[dy],
                           reinterpret_cast<char*>(dyRef),
                           m_tensorDescs[x],
                           reinterpret_cast<char*>(m_hostBuffers[x]),
                           m_tensorDescs[dw],
                           reinterpret_cast<char*>(m_hostBuffers[dw]),
                           dedwParams,
                           outIdx,
                           m_deviceType);
    ASSERT_TRUE(check) << "Error at: " << toString(outIdx, ',');
}
TEST_F_GC(SynGaudiSRAMSlicingTest, contiguous_reshape_removal)
{
    /* [SW-31184] */
    /*************
    * _spatial_convolution_n1077_0 node
    * inputs:
    *t1666[192, 16, 16, 8] (dtype=float32)
    *t201[1152, 192, 1, 1] (dtype=float32)
    * outputs:
    *t1669__0[1152, 16, 16, 8] (dtype=float32)
    *************/

    // create t1666 tensor
    unsigned t1666_max_sizes[] = {192, 16, 16, 8};
    unsigned t1666_min_sizes[] = {192, 16, 16, 8};
    unsigned t1666 = createTensors(1, INPUT_TENSOR, true, "t1666", MEM_INIT_ALL_ZERO, nullptr, t1666_max_sizes, 4,
                                   syn_type_single, nullptr, 0, 0, nullptr, false, t1666_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t201 tensor
    unsigned t201_max_sizes[] = {1152, 192, 1, 1};
    unsigned t201_min_sizes[] = {1152, 192, 1, 1};
    unsigned t201 = createTensors(1, INPUT_TENSOR, true, "t201", MEM_INIT_ALL_ZERO, nullptr, t201_max_sizes, 4,
                                  syn_type_single, nullptr, 0, 0, nullptr, false, t201_min_sizes,
                                  synTensorType::DATA_TENSOR)[0];

    // create t1669__0 tensor
    unsigned t1669_max_sizes[] = {1152, 16, 16, 8};
    unsigned t1669_min_sizes[] = {1152, 16, 16, 8};
    unsigned t1669__0 = createTensors(1, INPUT_TENSOR, true, "t1669__0", MEM_INIT_ALL_ZERO, nullptr, t1669_max_sizes, 4,
                                      syn_type_single, nullptr, 0, 0, nullptr, false, t1669_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    unsigned char _spatial_convolution_n1077_0_params[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                           0, 0, 178, 121, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 100, 109, 77,
                                                           127, 0, 0, 1, 0, 0, 0, 77, 127, 0, 0};
    addNodeToGraph("spatial_convolution", {t1666, t201}, {t1669__0}, (void*) _spatial_convolution_n1077_0_params, 88,
                   "_spatial_convolution_n1077_0");

    /*************
    * tpu_batch_normalization_moments_mean_reshape_n1078_0 node
    * inputs:
    *t1669__0[1152, 16, 16, 8] (dtype=float32)
    *t1672[1152, 2048, 1, 1] (dtype=uint32) (shape tensor)
    * outputs:
    *t1671[1152, 2048, 1, 1] (dtype=float32)
    *************/

    // create t1672 tensor
    unsigned t1672_max_sizes[] = {1152, 2048, 1, 1};
    unsigned t1672_min_sizes[] = {1152, 2048, 1, 1};
    unsigned t1672 = createTensors(1, INPUT_TENSOR, true, "t1672", MEM_INIT_ALL_ZERO, nullptr, t1672_max_sizes, 4,
                                   syn_type_uint32, nullptr, 0, 0, nullptr, false, t1672_min_sizes,
                                   synTensorType::SHAPE_TENSOR)[0];

    // create t1671 tensor
    unsigned t1671_max_sizes[] = {1152, 2048, 1, 1};
    unsigned t1671_min_sizes[] = {1152, 2048, 1, 1};
    unsigned t1671 = createTensors(1, INPUT_TENSOR, false, "t1671", MEM_INIT_ALL_ZERO, nullptr, t1671_max_sizes, 4,
                                   syn_type_single, nullptr, 0, 0, nullptr, false, t1671_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("reshape", {t1669__0, t1672}, {t1671}, nullptr, 0,
                   "tpu_batch_normalization_moments_mean_reshape_n1078_0");

    /*************
    * tpu_batch_normalization_moments_mean_reduce_mean_fwd_f32_n1079_0 node
    * inputs:
    *t1671[1152, 2048, 1, 1] (dtype=float32)
    * outputs:
    *t1670_0[1152, 1, 1, 1] (dtype=float32)
    *************/

    // create t1670_0 tensor
    unsigned t1670_max_sizes[] = {1152, 1, 1, 1};
    unsigned t1670_min_sizes[] = {1152, 1, 1, 1};
    unsigned t1670_0 = createTensors(1, INPUT_TENSOR, true, "t1670_0", MEM_INIT_ALL_ZERO, nullptr, t1670_max_sizes, 4,
                                     syn_type_single, nullptr, 0, 0, nullptr, false, t1670_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    unsigned char tpu_batch_normalization_moments_mean_reduce_mean_fwd_f32_n1079_0_params[] = {1, 0, 0, 0};
    addNodeToGraph("reduce_mean_fwd_f32", {t1671}, {t1670_0},
                   (void*) tpu_batch_normalization_moments_mean_reduce_mean_fwd_f32_n1079_0_params, 4,
                   "tpu_batch_normalization_moments_mean_reduce_mean_fwd_f32_n1079_0");

    /*************
    * tpu_batch_normalization_moments_Squeeze_reshape_n1080_0 node
    * inputs:
    *t1670_0[1152, 1, 1, 1] (dtype=float32)
    *t1674[1152] (dtype=uint32) (shape tensor)
    * outputs:
    *t1673_0[1152] (dtype=float32)
    *************/

    // create t1674 tensor
    unsigned t1674_max_sizes[] = {1152};
    unsigned t1674_min_sizes[] = {1152};
    unsigned t1674 = createTensors(1, INPUT_TENSOR, true, "t1674", MEM_INIT_ALL_ZERO, nullptr, t1674_max_sizes, 1,
                                   syn_type_uint32, nullptr, 0, 0, nullptr, false, t1674_min_sizes,
                                   synTensorType::SHAPE_TENSOR)[0];

    // create t1673_0 tensor
    unsigned t1673_max_sizes[] = {1152};
    unsigned t1673_min_sizes[] = {1152};
    unsigned t1673_0 = createTensors(1, INPUT_TENSOR, true, "t1673_0", MEM_INIT_ALL_ZERO, nullptr, t1673_max_sizes, 1,
                                     syn_type_single, nullptr, 0, 0, nullptr, false, t1673_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("reshape", {t1670_0, t1674}, {t1673_0}, nullptr, 0,
                   "tpu_batch_normalization_moments_Squeeze_reshape_n1080_0");

    /*************
    * tpu_batch_normalization_moments_SquaredDifference_sub_f32_n1081_0 node
    * inputs:
    *t1669__0[1152, 16, 16, 8] (dtype=float32)
    *t1670_0[1152, 1, 1, 1] (dtype=float32)
    * outputs:
    *t1676[1152, 16, 16, 8] (dtype=float32)
    *************/

    // create t1676 tensor
    unsigned t1676_max_sizes[] = {1152, 16, 16, 8};
    unsigned t1676_min_sizes[] = {1152, 16, 16, 8};
    unsigned t1676 = createTensors(1, INPUT_TENSOR, false, "t1676", MEM_INIT_ALL_ZERO, nullptr, t1676_max_sizes, 4,
                                   syn_type_single, nullptr, 0, 0, nullptr, false, t1676_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("sub_fwd_f32",
                   {t1669__0, t1670_0},
                   {t1676},
                   nullptr,
                   0,
                   "tpu_batch_normalization_moments_SquaredDifference_sub_f32_n1081_0");

    /*************
    * tpu_batch_normalization_moments_SquaredDifference_mult_f32_n1082_0 node
    * inputs:
    *t1676[1152, 16, 16, 8] (dtype=float32)
    *t1676[1152, 16, 16, 8] (dtype=float32)
    * outputs:
    *t1675_0[1152, 16, 16, 8] (dtype=float32)
    *************/

    // create t1675_0 tensor
    unsigned t1675_0_max_sizes[] = {1152, 16, 16, 8};
    unsigned t1675_0_min_sizes[] = {1152, 16, 16, 8};
    unsigned t1675_0 = createTensors(1, INPUT_TENSOR, false, "t1675_0", MEM_INIT_ALL_ZERO, nullptr, t1675_0_max_sizes,
                                     4, syn_type_single, nullptr, 0, 0, nullptr, false, t1675_0_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("mult_fwd_f32",
                   {t1676, t1676},
                   {t1675_0},
                   nullptr,
                   0,
                   "tpu_batch_normalization_moments_SquaredDifference_mult_f32_n1082_0");

    /*************
    * tpu_reshape_n1083_0 node
    * inputs:
    *t1675_0[1152, 16, 16, 8] (dtype=float32)
    *t1679[1152, 2048, 1, 1] (dtype=uint32) (shape tensor)
    * outputs:
    *t1678[1152, 2048, 1, 1] (dtype=float32)
    *************/

    // create t1679 tensor
    unsigned t1679_max_sizes[] = {1152, 2048, 1, 1};
    unsigned t1679_min_sizes[] = {1152, 2048, 1, 1};
    unsigned t1679 = createTensors(1, INPUT_TENSOR, true, "t1679", MEM_INIT_ALL_ZERO, nullptr, t1679_max_sizes, 4,
                                   syn_type_uint32, nullptr, 0, 0, nullptr, false, t1679_min_sizes,
                                   synTensorType::SHAPE_TENSOR)[0];

    // create t1678 tensor
    unsigned t1678_max_sizes[] = {1152, 2048, 1, 1};
    unsigned t1678_min_sizes[] = {1152, 2048, 1, 1};
    unsigned t1678 = createTensors(1, INPUT_TENSOR, false, "t1678", MEM_INIT_ALL_ZERO, nullptr, t1678_max_sizes, 4,
                                   syn_type_single, nullptr, 0, 0, nullptr, false, t1678_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("reshape", {t1675_0, t1679}, {t1678}, nullptr, 0, "tpu_reshape_n1083_0");

    /*************
    * tpu_reduce_mean_fwd_f32_n1084_0 node
    * inputs:
    *t1678[1152, 2048, 1, 1] (dtype=float32)
    * outputs:
    *t1677_0[1152, 1, 1, 1] (dtype=float32)
    *************/

    // create t1677_0 tensor
    unsigned t1677_0_max_sizes[] = {1152, 1, 1, 1};
    unsigned t1677_0_min_sizes[] = {1152, 1, 1, 1};
    unsigned t1677_0 = createTensors(1, INPUT_TENSOR, false, "t1677_0", MEM_INIT_ALL_ZERO, nullptr, t1677_0_max_sizes,
                                     4, syn_type_single, nullptr, 0, 0, nullptr, false, t1677_0_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    unsigned char tpu_reduce_mean_fwd_f32_n1084_0_params[] = {1, 0, 0, 0};
    addNodeToGraph("reduce_mean_fwd_f32", {t1678}, {t1677_0}, (void*) tpu_reduce_mean_fwd_f32_n1084_0_params, 4,
                   "tpu_reduce_mean_fwd_f32_n1084_0");

    /*************
    * tpu_batch_normalization_moments_Squeeze_1_reshape_n1085_0 node
    * inputs:
    *t1677_0[1152, 1, 1, 1] (dtype=float32)
    *t1681[1152] (dtype=uint32) (shape tensor)
    * outputs:
    *t1680_1_0[1152] (dtype=float32)
    *************/

    // create t1681 tensor
    unsigned t1681_max_sizes[] = {1152};
    unsigned t1681_min_sizes[] = {1152};
    unsigned t1681 = createTensors(1, INPUT_TENSOR, true, "t1681", MEM_INIT_ALL_ZERO, nullptr, t1681_max_sizes, 1,
                                   syn_type_uint32, nullptr, 0, 0, nullptr, false, t1681_min_sizes,
                                   synTensorType::SHAPE_TENSOR)[0];

    // create t1680_1_0 tensor
    unsigned t1680_1_0_max_sizes[] = {1152};
    unsigned t1680_1_0_min_sizes[] = {1152};
    unsigned t1680_1_0 = createTensors(1, INPUT_TENSOR, true, "t1680_1_0", MEM_INIT_ALL_ZERO, nullptr,
                                       t1680_1_0_max_sizes, 1, syn_type_single, nullptr, 0, 0, nullptr, false,
                                       t1680_1_0_min_sizes, synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("reshape", {t1677_0, t1681}, {t1680_1_0}, nullptr, 0,
                   "tpu_batch_normalization_moments_Squeeze_1_reshape_n1085_0");

    /*************
    * tpu_batch_normalization_batchnorm_add_add_f32_n1086_0 node
    * inputs:
    *t232_0[1] (dtype=float32)
    *t1680_1_0[1152] (dtype=float32)
    * outputs:
    *t1682_0[1152] (dtype=float32)
    *************/

    // create t232_0 tensor
    unsigned t232_0_max_sizes[] = {1};
    unsigned t232_0_min_sizes[] = {1};
    unsigned t232_0 = createTensors(1, INPUT_TENSOR, true, "t232_0", MEM_INIT_ALL_ZERO, nullptr, t232_0_max_sizes, 1,
                                    syn_type_single, nullptr, 0, 0, nullptr, false, t232_0_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];

    // create t1682_0 tensor
    unsigned t1682_0_max_sizes[] = {1152};
    unsigned t1682_0_min_sizes[] = {1152};
    unsigned t1682_0 = createTensors(1, INPUT_TENSOR, false, "t1682_0", MEM_INIT_ALL_ZERO, nullptr, t1682_0_max_sizes,
                                     1, syn_type_single, nullptr, 0, 0, nullptr, false, t1682_0_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("add_fwd_f32",
                   {t232_0, t1680_1_0},
                   {t1682_0},
                   nullptr,
                   0,
                   "tpu_batch_normalization_batchnorm_add_add_f32_n1086_0");

    /*************
    * tpu_batch_normalization_batchnorm_Rsqrt_rsqrt_fwd_f32_n1087_0 node
    * inputs:
    *t1682_0[1152] (dtype=float32)
    * outputs:
    *t1683_0[1152] (dtype=float32)
    *************/

    // create t1683_0 tensor
    unsigned t1683_0_max_sizes[] = {1152};
    unsigned t1683_0_min_sizes[] = {1152};
    unsigned t1683_0 = createTensors(1, INPUT_TENSOR, true, "t1683_0", MEM_INIT_ALL_ZERO, nullptr, t1683_0_max_sizes, 1,
                                     syn_type_single, nullptr, 0, 0, nullptr, false, t1683_0_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("rsqrt_fwd_f32", {t1682_0}, {t1683_0}, nullptr, 0,
                   "tpu_batch_normalization_batchnorm_Rsqrt_rsqrt_fwd_f32_n1087_0");

    /*************
    * tpu_batch_normalization_batchnorm_mul_mult_f32_n1088_0 node
    * inputs:
    *t1683_0[1152] (dtype=float32)
    *t202_0[1152] (dtype=float32)
    * outputs:
    *t1685_0[1152] (dtype=float32)
    *************/

    // create t202_0 tensor
    unsigned t202_0_max_sizes[] = {1152};
    unsigned t202_0_min_sizes[] = {1152};
    unsigned t202_0 = createTensors(1, INPUT_TENSOR, true, "t202_0", MEM_INIT_ALL_ZERO, nullptr, t202_0_max_sizes, 1,
                                    syn_type_single, nullptr, 0, 0, nullptr, false, t202_0_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];

    // create t1685_0 tensor
    unsigned t1685_0_max_sizes[] = {1152};
    unsigned t1685_0_min_sizes[] = {1152};
    unsigned t1685_0 = createTensors(1, INPUT_TENSOR, true, "t1685_0", MEM_INIT_ALL_ZERO, nullptr, t1685_0_max_sizes, 1,
                                     syn_type_single, nullptr, 0, 0, nullptr, false, t1685_0_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("mult_fwd_f32",
                   {t1683_0, t202_0},
                   {t1685_0},
                   nullptr,
                   0,
                   "tpu_batch_normalization_batchnorm_mul_mult_f32_n1088_0");

    /*************
    * tpu_batch_normalization_batchnorm_mul_1_reshape_n1089_0 node
    * inputs:
    *t1685_0[1152] (dtype=float32)
    *t1688[1152, 1, 1, 1] (dtype=uint32) (shape tensor)
    * outputs:
    *t1687[1152, 1, 1, 1] (dtype=float32)
    *************/

    // create t1688 tensor
    unsigned t1688_max_sizes[] = {1152, 1, 1, 1};
    unsigned t1688_min_sizes[] = {1152, 1, 1, 1};
    unsigned t1688 = createTensors(1, INPUT_TENSOR, true, "t1688", MEM_INIT_ALL_ZERO, nullptr, t1688_max_sizes, 4,
                                   syn_type_uint32, nullptr, 0, 0, nullptr, false, t1688_min_sizes,
                                   synTensorType::SHAPE_TENSOR)[0];

    // create t1687 tensor
    unsigned t1687_max_sizes[] = {1152, 1, 1, 1};
    unsigned t1687_min_sizes[] = {1152, 1, 1, 1};
    unsigned t1687 = createTensors(1, INPUT_TENSOR, false, "t1687", MEM_INIT_ALL_ZERO, nullptr, t1687_max_sizes, 4,
                                   syn_type_single, nullptr, 0, 0, nullptr, false, t1687_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("reshape", {t1685_0, t1688}, {t1687}, nullptr, 0,
                   "tpu_batch_normalization_batchnorm_mul_1_reshape_n1089_0");

    /*************
    * tpu_batch_normalization_batchnorm_mul_1_mult_f32_n1090_0 node
    * inputs:
    *t1669__0[1152, 16, 16, 8] (dtype=float32)
    *t1687[1152, 1, 1, 1] (dtype=float32)
    * outputs:
    *t1686_1_0[1152, 16, 16, 8] (dtype=float32)
    *************/

    // create t1686_1_0 tensor
    unsigned t1686_1_0_max_sizes[] = {1152, 16, 16, 8};
    unsigned t1686_1_0_min_sizes[] = {1152, 16, 16, 8};
    unsigned t1686_1_0 = createTensors(1, INPUT_TENSOR, true, "t1686_1_0", MEM_INIT_ALL_ZERO, nullptr,
                                       t1686_1_0_max_sizes, 4, syn_type_single, nullptr, 0, 0, nullptr, false,
                                       t1686_1_0_min_sizes, synTensorType::DATA_TENSOR)[0];
    addNodeToGraph("mult_fwd_f32",
                   {t1669__0, t1687},
                   {t1686_1_0},
                   nullptr,
                   0,
                   "tpu_batch_normalization_batchnorm_mul_1_mult_f32_n1090_0");

    compileTopology("graph_data");
}

TEST_F_GC(SynGaudiSRAMSlicingTest, gemm_broadcast_add_consumer_ASIC_CI)
{
    synDataType tensorType = syn_type_float;
    TestSizes   gemmAsize  = {768, 1024, 1, 1, 1};
    TestSizes   gemmBsize  = {1024, 768, 1, 1, 1};

    unsigned int gemmAInput = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,  // initializer
                                                  gemmAsize.data(),
                                                  2,
                                                  tensorType,
                                                  nullptr,  // strides
                                                  "gemmAInput");

    unsigned int gemmBInput = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,  // initializer
                                                  gemmBsize.data(),
                                                  2,
                                                  tensorType,
                                                  nullptr,  // strides
                                                  "gemmBInput");

    TestSizes    size       = {1024, 1024, 1, 1, 1};
    unsigned int gemmOutput = createTensors(1,
                                            OUTPUT_TENSOR,
                                            false,  // isPersistent
                                            "gemmOutput",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,  // initializer
                                            size.data(),
                                            2,
                                            tensorType)[0];

    synGEMMParams params;
    params.transpose_a = false;
    params.transpose_b = false;

    addNodeToGraph("gemm", {gemmAInput, gemmBInput}, {gemmOutput}, &params, sizeof(params), "gemm");

    unsigned int addFirstInput = connectOutputTensorToInputTensor(gemmOutput);

    TestSizes    secondAddSize  = {1024, 1, 1, 1, 1};
    unsigned int addSecondInput = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,  // initializer
                                                      secondAddSize.data(),
                                                      2,
                                                      tensorType,
                                                      nullptr,  // strides
                                                      "addSecondInput");

    unsigned int addOutput = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,  // initializer
                                                 size.data(),
                                                 2,
                                                 tensorType,
                                                 nullptr,  // strides
                                                 "addOutput");

    addNodeToGraph("add_fwd_f32", {addFirstInput, addSecondInput}, {addOutput}, nullptr, 0, "add");

    addConfigurationToRun(FIRST_RUN,
                          "SRAM_SLICER_MAX_CAPACITY_BYTES",
                          GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.getValueStr());

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({addOutput});
}

TEST_F_GC(SynGaudiSRAMSlicingTest, gemm_with_int64_producer_in_chain_ASIC_CI, {synDeviceGaudi2, synDeviceGaudi3})
{
    constexpr unsigned         dims  = 3;
    std::array<unsigned, dims> sizes = {256, 256, 384};

    unsigned int firstCastInput = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_ALL_ZERO,
                                                      nullptr,  // initializer
                                                      sizes.data(),
                                                      sizes.size(),
                                                      syn_type_int64,
                                                      nullptr,  // strides
                                                      "firstCastInput");

    ns_ConstantKernel::Params constParams = {5};

    addNodeToGraph("constant_i64", {}, {firstCastInput}, &constParams, sizeof(constParams), "constant_i64");

    unsigned int firstCastOutput = createTensor(INPUT_TENSOR,
                                                MEM_INIT_ALL_ONES,
                                                nullptr,  // initializer
                                                sizes.data(),
                                                sizes.size(),
                                                syn_type_int32,
                                                nullptr  // strides
    );

    addNodeToGraph("cast_i64_to_i32", {firstCastInput}, {firstCastOutput}, nullptr, 0, "cast_i64_to_i32");

    unsigned int gemmAInput = createTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,  // initializer
                                           sizes.data(),
                                           sizes.size(),
                                           syn_type_float,
                                           nullptr  // strides
    );

    addNodeToGraph("cast_i32_to_f32", {firstCastOutput}, {gemmAInput}, nullptr, 0, "cast_i32_to_f32");

    unsigned int gemmBInput = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,  // initializer
                                                  sizes.data(),
                                                  sizes.size(),
                                                  syn_type_float,
                                                  nullptr,  // strides
                                                  "gemmBInput");

    unsigned int gemmOutput = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,  // initializer
                                                  sizes.data(),
                                                  sizes.size(),
                                                  syn_type_float,
                                                  nullptr,  // strides
                                                  "gemmOutput");

    synGEMMParams params;
    params.transpose_a = false;
    params.transpose_b = false;

    addNodeToGraph("batch_gemm", {gemmAInput, gemmBInput}, {gemmOutput}, &params, sizeof(params), "gemm");

    addConfigurationToRun(FIRST_RUN,
                          "SRAM_SLICER_MAX_CAPACITY_BYTES",
                          GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.getValueStr());

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({gemmOutput});
}