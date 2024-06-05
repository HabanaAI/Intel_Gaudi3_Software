#include <vector>
#include "node_factory.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "test_utils.h"
#include "utils.h"
#include <algorithm>

typedef std::tuple<synDataType, synDataType, synDataType> dataTypeTuple;     // for easy param get
typedef std::vector<double>                               scaleVector;       // for easy compare against test result
typedef std::vector<scaleVector>                          scaleVectorPerIO;  // for easy compare against test result

class injectScaleForMMENodesTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<dataTypeTuple, scaleVector, scaleVector>>
{
protected:
    void init();

    void runTest(HabanaGraph& g);

    bool compareScales(TensorPtr tensor, double scale);

    synDataType inputType   = syn_type_na;
    synDataType weightsType = syn_type_na;
    synDataType outputType  = syn_type_na;

    std::vector<TSize> inputSizes   = {};
    std::vector<TSize> weightsSizes = {};
    std::vector<TSize> outSizes     = {};

    TSize inputElementsNum   = 0;
    TSize weightsElementsNum = 0;
    TSize outputElementsNum  = 0;

    // We just need below data vectors for creating the tensors, don't care about vector type.
    std::vector<float> inputData       = {};
    std::vector<float> weightsData     = {};
    std::vector<float> inputDataCast   = {};
    std::vector<float> weightsDataCast = {};
    std::vector<float> biasData        = {};

    synConvolutionParams params;
};

void injectScaleForMMENodesTest::init()
{
    const unsigned N = 1;
    const unsigned H = 4;
    const unsigned W = H;
    const unsigned C = 1;
    inputSizes       = {C, W, H, N};
    inputElementsNum = multiplyElements(inputSizes.begin(), inputSizes.end());
    inputData.resize(inputElementsNum);

    const unsigned weightsDimSize = 1;
    const unsigned weightsNum     = 1;
    const unsigned weightsStride  = 1;
    const unsigned weightsPadding = 0;
    weightsSizes                  = {weightsNum, C, weightsDimSize, weightsDimSize};
    weightsElementsNum            = multiplyElements(weightsSizes.begin(), weightsSizes.end());
    weightsData.resize(weightsElementsNum);

    const unsigned outW = ((W - weightsDimSize + 2 * weightsPadding) / weightsStride) + 1;
    const unsigned outH = ((H - weightsDimSize + 2 * weightsPadding) / weightsStride) + 1;
    const unsigned outC = weightsNum;
    outSizes            = {outC, outW, outH, N};
    outputElementsNum   = multiplyElements(outSizes.begin(), outSizes.end());

    params.kH   = weightsDimSize;
    params.kW   = weightsDimSize;
    params.padT = weightsPadding;
    params.padL = weightsPadding;
    params.dW   = weightsStride;
    params.dH   = weightsStride;
}

bool injectScaleForMMENodesTest::compareScales(TensorPtr tensor, double scale)
{
    switch (tensor->getElementType())
    {
        case syn_type_bf16:
        {
            bf16_t* data = reinterpret_cast<bf16_t*>(tensor->getData());
            return data[0] == (bf16_t)(float)scale;
        }
        case syn_type_fp16:
        {
            fp16_t* data = reinterpret_cast<fp16_t*>(tensor->getData());
            return data[0] == (fp16_t)(float)scale;
        }
        case syn_type_float:
        {
            float* data = reinterpret_cast<float*>(tensor->getData());
            return data[0] == (float)scale;
        }
        default:
        {
            LOG_WARN(QUANT, "Unsupported buffer data type for tensor {}", tensor->getName());
            return false;
        }
    }
}

void injectScaleForMMENodesTest::runTest(HabanaGraph& g)
{
    std::tie(inputType, weightsType, outputType) = std::get<0>(GetParam());
    const scaleVector& scales                    = std::get<1>(GetParam());
    const scaleVector& expectedScales            = std::get<2>(GetParam());

    TensorPtr castInputForMMEInput =
        TensorPtr(new Tensor(4U, inputSizes.data(), inputType, reinterpret_cast<char*>(inputDataCast.data())));
    castInputForMMEInput->setName("castInputForMMEInput", true);
    TensorPtr castInputForMMEWeights =
        TensorPtr(new Tensor(4U, weightsSizes.data(), weightsType, reinterpret_cast<char*>(weightsDataCast.data())));
    castInputForMMEWeights->setName("castInputForMMEWeights", true);
    TensorPtr inputTensor =
        TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_fp8_143, reinterpret_cast<char*>(inputData.data())));
    inputTensor->setName("inputTensor", true);
    TensorPtr weightsTensor =
        TensorPtr(new Tensor(4U, weightsSizes.data(), syn_type_fp8_143, reinterpret_cast<char*>(weightsData.data())));
    weightsTensor->setName("weightsTensor", true);

    weightsTensor->setAsStaticParam(true);
    weightsTensor->setAsWeights();

    TensorPtr outputTensor = TensorPtr(new Tensor(4U, outSizes.data(), outputType));
    outputTensor->setName("outputTensor", true);

    NodePtr castInput =
        NodeFactory::createNode({castInputForMMEInput},
                                {inputTensor},
                                nullptr,
                                getCastGUID(castInputForMMEInput->getElementType(), inputTensor->getElementType()),
                                "cast_input");
    ASSERT_TRUE(GraphEditor::addNode(g, castInput));

    NodePtr castWeights =
        NodeFactory::createNode({castInputForMMEWeights},
                                {weightsTensor},
                                nullptr,
                                getCastGUID(castInputForMMEWeights->getElementType(), weightsTensor->getElementType()),
                                "cast_weights");
    ASSERT_TRUE(GraphEditor::addNode(g, castWeights));

    NodePtr conv = NodeFactory::createNode({inputTensor, weightsTensor},
                                           {outputTensor},
                                           &params,
                                           NodeFactory::convolutionNodeTypeName,
                                           "conv_node");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    QuantizationData quantDataMMeInput(syn_type_fp8_143);
    quantDataMMeInput.setScale(scales.at(0));

    QuantizationData quantDataMMeWeights(syn_type_fp8_143);
    quantDataMMeWeights.setScale(scales.at(1));

    inputTensor->setQuantizationParams(quantDataMMeInput);
    weightsTensor->setQuantizationParams(quantDataMMeWeights);

    int countExpectedScales =
        std::count_if(expectedScales.begin(), expectedScales.end(), [](int i) { return i != -1; });
    unsigned origNumElem = g.getNodes().size();

    ASSERT_TRUE(injectScaleForMMENodes(g));
    auto nodes = g.getNodes();

    ASSERT_EQ(origNumElem + countExpectedScales, nodes.size());

    for (const NodePtr& node : nodes)
    {
        if (node->getNodeName() == "cast_input_scale")
        {
            double expectedScale = expectedScales.at(0);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        if (node->getNodeName() == "cast_weights_scale")
        {
            double expectedScale = expectedScales.at(1);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        if (node->getNodeName() == "conv_node_descale_0")
        {
            double expectedScale = expectedScales.at(2);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
    }
}

TEST_P(injectScaleForMMENodesTest, scale_mme_nodes_test_gaudi2)
{
    init();

    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    runTest(g);
}

INSTANTIATE_TEST_SUITE_P(,
                         injectScaleForMMENodesTest,
                         testing::Values(std::make_tuple(dataTypeTuple(syn_type_float, syn_type_float, syn_type_float),
                                                         scaleVector {2, 3},      // input scale values
                                                         scaleVector {2, 3, 6}),  // expected scales
                                         std::make_tuple(dataTypeTuple(syn_type_bf16, syn_type_bf16, syn_type_bf16),
                                                         scaleVector {2, 2},
                                                         scaleVector {2, 2, 4}),
                                         std::make_tuple(dataTypeTuple(syn_type_fp16, syn_type_fp16, syn_type_fp16),
                                                         scaleVector {2, 2},
                                                         scaleVector {2, 2, 4}),
                                         // should not scale int
                                         std::make_tuple(dataTypeTuple(syn_type_int32, syn_type_int32, syn_type_int32),
                                                         scaleVector {2, 2},
                                                         scaleVector {-1, -1, -1}),
                                         // should scale only 1 tensor and descale
                                         std::make_tuple(dataTypeTuple(syn_type_float, syn_type_int32, syn_type_float),
                                                         scaleVector {2, 2},
                                                         scaleVector {2, -1, 2})

                                             )

);

// Test input:
//      * conv node with non-default scales for its inputs
//      * fp8_gemm_f32 complex guid node with non-default scales for its inputs
// Expected compilation behavior:
//  1. first run of injectScaleForMMENodes will apply scaling/descaling for the conv node since it defined as MME node
//     conv input scales will be set to default value (1.0) after adding the scaling nodes with the original input scales
//  2. cguid pass will replace the fp8_gemm_f32 with gemm node
//  3. second run of injectScaleForMMENodes will add the scaling/descaling nodes with the input scale values, and updating
//     the gemm inputs scales to default value (1.0)
TEST_F(injectScaleForMMENodesTest, inject_scales_for_cguid)
{
    init();

    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    const scaleVector scales         = {2, 3, 4, 5};
    const scaleVector expectedScales = {2, 3, 6, 4, 5, 20};

    setGlobalConfForTest(GCFG_UPDATE_GRAPH_OUTPUT_MME, "true");
    setGlobalConfForTest(GCFG_ENABLE_CALC_DYNAMIC_RANGE, "true");

    TensorPtr inputTensor =
        TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_float, reinterpret_cast<char*>(inputData.data())));

    inputTensor->setName("convInputTensor", true);
    synMemoryDescriptor inputTensor_memDesc(true);
    inputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    inputTensor->setMemoryDescriptor(inputTensor_memDesc);

    QuantizationData quantDataMMeInput(syn_type_fp8_143);
    quantDataMMeInput.setScale(scales.at(0));
    quantDataMMeInput.m_isUserQuantInfo = true;
    inputTensor->setQuantizationParams(quantDataMMeInput);

    TensorPtr weightsTensor =
        TensorPtr(new Tensor(4U, weightsSizes.data(), syn_type_float, reinterpret_cast<char*>(weightsData.data())));

    weightsTensor->setName("convWeightsTensor", true);
    synMemoryDescriptor weightsTensor_memDesc(true);
    weightsTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    weightsTensor->setMemoryDescriptor(weightsTensor_memDesc);
    weightsTensor->setAsStaticParam();

    QuantizationData quantDataMMeWeights(syn_type_fp8_143);
    quantDataMMeWeights.setScale(scales.at(1));
    quantDataMMeWeights.m_isUserQuantInfo = true;
    weightsTensor->setQuantizationParams(quantDataMMeWeights);

    TensorPtr convOutputTensor = TensorPtr(new Tensor(4U, outSizes.data(), syn_type_float));
    convOutputTensor->setName("convOutputTensor", true);
    synMemoryDescriptor convOutputTensor_memDesc(true);
    convOutputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    convOutputTensor->setMemoryDescriptor(convOutputTensor_memDesc);

    NodePtr conv = NodeFactory::createNode({inputTensor, weightsTensor},
                                           {convOutputTensor},
                                           &params,
                                           NodeFactory::convolutionNodeTypeName,
                                           "conv_node");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    std::vector<TSize> gemmInputSizes   = {3, 6};
    std::vector<TSize> gemmWeightsSizes = {6, 4};
    std::vector<TSize> gemmOutputSizes  = {3, 4};

    std::vector<float> gemmInputData(18);
    std::vector<float> gemmWeightsData(24);

    TensorPtr gemmInputTensor =
        TensorPtr(new Tensor(2U, gemmInputSizes.data(), syn_type_float, reinterpret_cast<char*>(gemmInputData.data())));

    gemmInputTensor->setName("gemmInputTensor", true);
    synMemoryDescriptor gemmInputTensor_memDesc(true);
    gemmInputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    gemmInputTensor->setMemoryDescriptor(gemmInputTensor_memDesc);

    TensorPtr gemmWeightsTensor =
        TensorPtr(new Tensor(2U, gemmWeightsSizes.data(), syn_type_float, reinterpret_cast<char*>(gemmWeightsData.data())));

    gemmWeightsTensor->setName("gemmWeightsTensor", true);
    synMemoryDescriptor gemmWeightsTensor_memDesc(true);
    gemmWeightsTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    gemmWeightsTensor->setMemoryDescriptor(gemmWeightsTensor_memDesc);
    gemmWeightsTensor->setAsStaticParam();

    TensorPtr outputTensor = TensorPtr(new Tensor(2U, gemmOutputSizes.data(), syn_type_float));
    outputTensor->setName("GemmOutputTensor", true);
    synMemoryDescriptor outputTensor_memDesc(true);
    outputTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    outputTensor->setMemoryDescriptor(outputTensor_memDesc);

    TensorPtr castOutputForGemmInput =
        TensorPtr(new Tensor(2U, gemmInputSizes.data(), syn_type_fp8_143));
    castOutputForGemmInput->setName("castOutputForGemmInput", true);

    QuantizationData quantDataGemmInput(syn_type_fp8_143);
    quantDataGemmInput.setScale(scales.at(2));
    quantDataGemmInput.m_isUserQuantInfo = true;
    castOutputForGemmInput->setQuantizationParams(quantDataGemmInput);

    TensorPtr castOutputForGemmWeights =
        TensorPtr(new Tensor(2U, gemmWeightsSizes.data(), syn_type_fp8_143));
    castOutputForGemmWeights->setName("castOutputForGemmWeights", true);

    QuantizationData quantDataGemmWeights(syn_type_fp8_143);
    quantDataGemmWeights.setScale(scales.at(3));
    quantDataGemmWeights.m_isUserQuantInfo = true;
    castOutputForGemmWeights->setQuantizationParams(quantDataGemmWeights);

    NodePtr castInput =
        NodeFactory::createNode({gemmInputTensor},
                                {castOutputForGemmInput},
                                nullptr,
                                getCastGUID(gemmInputTensor->getElementType(), castOutputForGemmInput->getElementType()),
                                "cast_input");
    ASSERT_TRUE(GraphEditor::addNode(g, castInput));

    NodePtr castWeights =
        NodeFactory::createNode({gemmWeightsTensor},
                                {castOutputForGemmWeights},
                                nullptr,
                                getCastGUID(gemmWeightsTensor->getElementType(), castOutputForGemmWeights->getElementType()),
                                "cast_weights");
    ASSERT_TRUE(GraphEditor::addNode(g, castWeights));

    synGEMMParams gemmParams;
    gemmParams.transpose_a = false;
    gemmParams.transpose_b = false;
    NodePtr gemm = NodeFactory::createNode({castOutputForGemmInput, castOutputForGemmWeights},
                                           {outputTensor},
                                           &gemmParams,
                                           "fp8_gemm_f32",
                                           "fp8_gemm_f32");
    ASSERT_TRUE(GraphEditor::addNode(g, gemm));

    ASSERT_TRUE(g.compile());
    auto nodes = g.getNodes();

    int mmeCounter  = 0;
    int multCounter = 0;
    int divCounter  = 0;
    for (const NodePtr& node : nodes)
    {
        if (g.runsOnMME(node)) mmeCounter++;
        if (node->getNodeTypeStr() == "mult_fwd_f32") multCounter++;
        if (node->getNodeTypeStr() == "div_fwd_f32")  divCounter++;

        if (node->getNodeName() == "convInputTensor_cast_scale_complex/mult_fwd_bf16_1_optimized")
        {
            double expectedScale = expectedScales.at(0);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        else if (node->getNodeName() == "convWeightsTensor_cast_scale_complex/mult_fwd_bf16_0")
        {
            double expectedScale = expectedScales.at(1);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        else if (node->getNodeName() == "conv_node_descale_0_complex/div_fwd_bf16_0_optimized")
        {
            double expectedScale = expectedScales.at(2);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        else if (node->getNodeName() == "conv_node")
        {
            ASSERT_EQ(node->getInput(0)->getScale(), 1.0);
            ASSERT_EQ(node->getInput(1)->getScale(), 1.0);
        }
        else if (node->getNodeName() == "cast_input_scale_complex/mult_fwd_f32_0_optimized")
        {
            double expectedScale = expectedScales.at(3);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        else if (node->getNodeName() == "cast_weights_scale_complex/mult_fwd_f32_0")
        {
            double expectedScale = expectedScales.at(4);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        else if (node->getNodeName() == "fp8_gemm_f32_complex/gemm_0_descale_0_complex/div_fwd_f32_0")
        {
            double expectedScale = expectedScales.at(5);
            ASSERT_TRUE(compareScales(node->getInput(1), expectedScale));
        }
        else if (node->getNodeName() == "fp8_gemm_f32_complex/gemm_0")
        {
            ASSERT_EQ(node->getInput(0)->getScale(), 1.0);
            ASSERT_EQ(node->getInput(1)->getScale(), 1.0);
        }
    }
    const int expectedMMENodes = 2;
    ASSERT_EQ(mmeCounter,  expectedMMENodes);
    ASSERT_EQ(divCounter, expectedMMENodes * 2);
    ASSERT_EQ(multCounter,  expectedMMENodes);
}

class injectPCScaleForMMENodesTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<dataTypeTuple, scaleVectorPerIO, scaleVectorPerIO>>
{
protected:
    void init();

    void runTest(HabanaGraph& g);

    template<class T>
    bool compareBuffer(TensorPtr tensor, const scaleVector& scales);

    bool compareScales(TensorPtr tensor, const scaleVector& scales);

    synDataType inputType   = syn_type_na;
    synDataType weightsType = syn_type_na;
    synDataType outputType  = syn_type_na;

    std::vector<TSize> inputSizes   = {};
    std::vector<TSize> weightsSizes = {};
    std::vector<TSize> outSizes     = {};

    TSize inputElementsNum   = 0;
    TSize weightsElementsNum = 0;
    TSize outputElementsNum  = 0;

    // We just need below data vectors for creating the tensors, don't care about vector type.
    std::vector<float> inputData       = {};
    std::vector<float> weightsData     = {};
    std::vector<float> inputDataCast   = {};
    std::vector<float> weightsDataCast = {};
    std::vector<float> biasData        = {};

    synConvolutionParams params;
};

void injectPCScaleForMMENodesTest::init()
{
    const unsigned N = 1;
    const unsigned H = 4;
    const unsigned W = H;
    const unsigned C = 3;
    inputSizes       = {C, W, H, N};
    inputElementsNum = multiplyElements(inputSizes.begin(), inputSizes.end());
    inputData.resize(inputElementsNum);

    const unsigned weightsDimSize = 2;
    const unsigned weightsNum     = 3;
    const unsigned weightsStride  = 1;
    const unsigned weightsPadding = 0;
    weightsSizes                  = {weightsNum, C, weightsDimSize, weightsDimSize};
    weightsElementsNum            = multiplyElements(weightsSizes.begin(), weightsSizes.end());
    weightsData.resize(weightsElementsNum);

    const unsigned outW = ((W - weightsDimSize + 2 * weightsPadding) / weightsStride) + 1;
    const unsigned outH = ((H - weightsDimSize + 2 * weightsPadding) / weightsStride) + 1;
    const unsigned outC = weightsNum;
    outSizes            = {outC, outW, outH, N};
    outputElementsNum   = multiplyElements(outSizes.begin(), outSizes.end());

    params.kH   = weightsDimSize;
    params.kW   = weightsDimSize;
    params.padT = weightsPadding;
    params.padL = weightsPadding;
    params.dW   = weightsStride;
    params.dH   = weightsStride;
}

template<class T>
bool injectPCScaleForMMENodesTest::compareBuffer(TensorPtr tensor, const scaleVector& scales)
{
    std::vector<T> castedVector;
    for (const auto& scale : scales)
    {
        T castedVal((float)scale);
        castedVector.push_back(castedVal);
    }

    if (scales.size() != tensor->getTotalElements()) return false;

    T* data = reinterpret_cast<T*>(tensor->getData());
    for (auto i = 0; i < castedVector.size(); ++i)
    {
        if (castedVector[i] != data[i]) return false;
    }
    return true;
}

bool injectPCScaleForMMENodesTest::compareScales(TensorPtr tensor, const scaleVector& scales)
{
    switch (tensor->getElementType())
    {
        case syn_type_bf16:
        {
            return compareBuffer<bf16_t>(tensor, scales);
        }
        case syn_type_fp16:
        {
            return compareBuffer<fp16_t>(tensor, scales);
        }
        case syn_type_float:
        {
            return compareBuffer<float>(tensor, scales);
        }
        default:
        {
            LOG_WARN(QUANT, "Unsupported buffer data type for tensor {}", tensor->getName());
            return false;
        }
    }
}

void injectPCScaleForMMENodesTest::runTest(HabanaGraph& g)
{
    std::tie(inputType, weightsType, outputType) = std::get<0>(GetParam());
    const scaleVectorPerIO& scales               = std::get<1>(GetParam());
    const scaleVectorPerIO& expectedScalesPerIO  = std::get<2>(GetParam());

    TensorPtr castInputForMMEInput =
        TensorPtr(new Tensor(4U, inputSizes.data(), inputType, reinterpret_cast<char*>(inputDataCast.data())));
    castInputForMMEInput->setName("castInputForMMEInput", true);
    TensorPtr castInputForMMEWeights =
        TensorPtr(new Tensor(4U, weightsSizes.data(), weightsType, reinterpret_cast<char*>(weightsDataCast.data())));
    castInputForMMEWeights->setName("castInputForMMEWeights", true);
    TensorPtr inputTensor =
        TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_fp8_143, reinterpret_cast<char*>(inputData.data())));
    inputTensor->setName("inputTensor", true);
    TensorPtr weightsTensor =
        TensorPtr(new Tensor(4U, weightsSizes.data(), syn_type_fp8_143, reinterpret_cast<char*>(weightsData.data())));
    weightsTensor->setName("weightsTensor", true);

    weightsTensor->setAsStaticParam(true);
    weightsTensor->setAsWeights();

    TensorPtr outputTensor = TensorPtr(new Tensor(4U, outSizes.data(), outputType));
    outputTensor->setName("outputTensor", true);

    NodePtr castInput =
        NodeFactory::createNode({castInputForMMEInput},
                                {inputTensor},
                                nullptr,
                                getCastGUID(castInputForMMEInput->getElementType(), inputTensor->getElementType()),
                                "cast_input");
    ASSERT_TRUE(GraphEditor::addNode(g, castInput));

    NodePtr castWeights =
        NodeFactory::createNode({castInputForMMEWeights},
                                {weightsTensor},
                                nullptr,
                                getCastGUID(castInputForMMEWeights->getElementType(), weightsTensor->getElementType()),
                                "cast_weights");
    ASSERT_TRUE(GraphEditor::addNode(g, castWeights));

    NodePtr conv = NodeFactory::createNode({inputTensor, weightsTensor},
                                           {outputTensor},
                                           &params,
                                           NodeFactory::convolutionNodeTypeName,
                                           "conv_node");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    QuantizationData quantDataMMeInput(syn_type_fp8_143);
    quantDataMMeInput.setScale(scales[0][0]);

    QuantizationData quantDataMMeWeights(syn_type_fp8_143);
    quantDataMMeWeights.reset(3, syn_type_fp8_143);
    quantDataMMeWeights.setScale(scales[1][0], 0);
    quantDataMMeWeights.setScale(scales[1][1], 1);
    quantDataMMeWeights.setScale(scales[1][2], 2);

    inputTensor->setQuantizationParams(quantDataMMeInput);
    weightsTensor->setQuantizationParams(quantDataMMeWeights);
    weightsTensor->setPerChannelQuant(true, true);

    int countExpectedScales =
        std::count_if(expectedScalesPerIO.begin(), expectedScalesPerIO.end(), [](const scaleVector& scales) {
            return std::all_of(scales.begin(), scales.end(), [](const double& scale) { return scale != -1; });
        });
    unsigned origNumElem = g.getNodes().size();

    ASSERT_TRUE(injectScaleForMMENodes(g));
    auto nodes = g.getNodes();

    ASSERT_EQ(origNumElem + countExpectedScales, nodes.size());

    for (const NodePtr& node : nodes)
    {
        scaleVector expectedScales;
        if (node->getNodeName() == "cast_input_scale")
        {
            expectedScales = expectedScalesPerIO[0];
        }
        else if (node->getNodeName() == "cast_weights_scale")
        {
            expectedScales = expectedScalesPerIO[1];
        }
        else if (node->getNodeName() == "conv_node_descale_0")
        {
            expectedScales = expectedScalesPerIO[2];
        }
        else
        {
            continue;
        }
        ASSERT_TRUE(compareScales(node->getInput(1), expectedScales));
    }
}

TEST_P(injectPCScaleForMMENodesTest, scale_mme_nodes_test_gaudi2)
{
    init();

    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    runTest(g);
}

INSTANTIATE_TEST_SUITE_P(,
                         injectPCScaleForMMENodesTest,
                         testing::Values(std::make_tuple(dataTypeTuple(syn_type_float, syn_type_float, syn_type_float),
                                                         scaleVectorPerIO {{2}, {3, 7, 2}},               // input scale values
                                                         scaleVectorPerIO {{2}, {3, 7, 2}, {6, 14, 4}}),  // expected scales
                                         std::make_tuple(dataTypeTuple(syn_type_bf16, syn_type_bf16, syn_type_bf16),
                                                         scaleVectorPerIO {{2}, {3, 7, 2}},
                                                         scaleVectorPerIO {{2}, {3, 7, 2}, {6, 14, 4}}),
                                         std::make_tuple(dataTypeTuple(syn_type_fp16, syn_type_fp16, syn_type_fp16),
                                                         scaleVectorPerIO {{2}, {3, 7, 2}},
                                                         scaleVectorPerIO {{2}, {3, 7, 2}, {6, 14, 4}}),
                                         // should not scale int
                                         std::make_tuple(dataTypeTuple(syn_type_int32, syn_type_int32, syn_type_int32),
                                                         scaleVectorPerIO {{2}, {3, 7, 2}},
                                                         scaleVectorPerIO {{-1}, {-1}, {-1}}),
                                         // should scale only 1 tensor and descale
                                         std::make_tuple(dataTypeTuple(syn_type_float, syn_type_int32, syn_type_float),
                                                         scaleVectorPerIO {{2}, {3, 7, 2}},
                                                         scaleVectorPerIO {{2}, {-1}, {2}})

                                             )

);
