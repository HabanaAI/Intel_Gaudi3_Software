#include "generic_graph_test.h"
#include "node_factory.h"
#include "node_utils.h"
#include "fuse_convert_mme.cpp"

class FuseConvertMmeTest : public GenericGraphTest
{
    protected:
    virtual void SetUp() override
    {
        GenericGraphTest::SetUp();
    }
};

void runFuseConvertMmeTest(HabanaGraph& m_graph, float scale)
{
    SizeArray inputSize = {5, 5};
    SizeArray scaleSize = {1, 1};
    std::vector<float> scaleDataGemm = {1};
    std::vector<float> scaleDataConvert = {scale};

    char* scaleBufferConvert = reinterpret_cast<char*>(scaleDataConvert.data());
    char* scaleBufferMmeInput = reinterpret_cast<char*>(scaleDataGemm.data());
    char* scaleBufferMmeWeight = reinterpret_cast<char*>(scaleDataGemm.data());
    char* scaleBufferMmeOutput = reinterpret_cast<char*>(scaleDataGemm.data());

    TensorPtr t1 = std::make_shared<Tensor>(2U, inputSize.data(), syn_type_fp8_143);
    TensorPtr t2 = std::make_shared<Tensor>(2U, inputSize.data(), syn_type_fp8_143);
    TensorPtr t3 = std::make_shared<Tensor>(2U, inputSize.data(), syn_type_bf16);
    TensorPtr t4 = std::make_shared<Tensor>(2U, inputSize.data(), syn_type_fp8_143);

    TensorPtr scaleConvert = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scaleConvert->setAsStaticParam(true);
    scaleConvert->setTensorBuffer(scaleBufferConvert, 1 * sizeof(float), syn_type_float);
    TensorPtr scaleMmeInput = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scaleMmeInput->setAsStaticParam(true);
    scaleMmeInput->setTensorBuffer(scaleBufferMmeInput, 1 * sizeof(float), syn_type_float);
    TensorPtr scaleMmeWeight = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scaleMmeWeight->setAsStaticParam(true);
    scaleMmeWeight->setTensorBuffer(scaleBufferMmeWeight, 1 * sizeof(float), syn_type_float);
    TensorPtr scaleMmeOuput = std::make_shared<Tensor>(1U, scaleSize.data(), syn_type_float);
    scaleMmeOuput->setAsStaticParam(true);
    scaleMmeOuput->setTensorBuffer(scaleBufferMmeOutput, 1 * sizeof(float), syn_type_float);

    NodePtr   gemmFp8ToBf16         = NodeFactory::createNode({t1, t2, scaleMmeInput,
                                                                 scaleMmeWeight, nullptr,
                                                                 nullptr, scaleMmeOuput},
                                                                 {t3}, nullptr,
                                                                 "fp8_gemm_bf16",
                                                                 "gemm_fp8_to_bf16");
    NodePtr   convertFromBf16ToFp8 = NodeFactory::createNode({t3, scaleConvert}, {t4}, nullptr,
                                                                 "convert_to_fp8_bf16",
                                                                 "convert_from_bf16_to_fp8");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, gemmFp8ToBf16))
                << "Failed to add gemm_fp8_to_bf16 node to graph";

    ASSERT_TRUE(GraphEditor::addNode(m_graph, convertFromBf16ToFp8))
                << "Failed to add convert_from_bf16_to_fp8 node to graph";

    m_graph.setInferenceMode(true);

    fuseConvertMme(m_graph);

    TensorPtr scaleOutputTensor = gemmFp8ToBf16->getInput(FP8_GEMM_INV_SCALE_OUT_IDX);
    void* data = scaleOutputTensor->getAddress();
    float scaleOutput = *(reinterpret_cast<float*>(data));
    ASSERT_EQ(scaleOutput, scale);

    ASSERT_EQ(m_graph.getNodes().size(), 1);

    for(auto node : m_graph.getNodes())
    {
        ASSERT_EQ(isConvertFp8Node(node), false);
        ASSERT_EQ(isFp8MmeCguid(node), true);
    }
}

TEST_P(FuseConvertMmeTest, fuse_convert_to_mme)
{
    runFuseConvertMmeTest(*m_graph, 1);
}
TEST_P(FuseConvertMmeTest, fuse_convert_to_mme_scale_replacement)
{
    runFuseConvertMmeTest(*m_graph, 16);
}

TEST_P(FuseConvertMmeTest, fuse_convert_conv)
{
    const TSize H = 5;
    const TSize W = 5;
    const TSize C = 2;
    const TSize N = 1;
    const TSize K = 2;
    const TSize R = 2;
    const TSize S = 2;

    const std::vector<TSize> inputSizes = {C, W, H, N};
    const std::vector<TSize> weightsSizes = {K, C, S, R};

    const TSize weightsStride  = 1;
    const TSize weightsPadding = 1;
    const TSize outW = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC = K;
    //                                     C    W    H    N
    const std::vector<TSize> outSizes  = {outC, outW, outH, N};

    SizeArray scaleSize                 = {1};
    std::vector<float> convScaleData    = {0.0625}; // HW aligned scale (1/16)
    char* convScaleBuffer               = reinterpret_cast<char*>(convScaleData.data());

    pTensor convIn      = pTensor(new Tensor(4U, inputSizes.data(), syn_type_fp8_143));
    pTensor convWeight  = pTensor(new Tensor(4U, weightsSizes.data(), syn_type_fp8_143));
    pTensor inScale     = pTensor(new Tensor(1U, scaleSize.data(), syn_type_float));
    pTensor weightScale = pTensor(new Tensor(1U, scaleSize.data(), syn_type_float));
    pTensor convOut     = pTensor(new Tensor(4U, outSizes.data(), syn_type_bf16));

    pTensor convertOut   = pTensor(new Tensor(4U, outSizes.data(), syn_type_fp8_143));
    pTensor convertScale = pTensor(new Tensor(1U, scaleSize.data(), syn_type_float));

    convWeight->setAsStaticParam(true);
    inScale->setAsStaticParam(true);
    weightScale->setAsStaticParam(true);
    convertScale->setAsStaticParam(true);
    inScale->setTensorBuffer(convScaleBuffer, 1 * sizeof(float), syn_type_float);
    weightScale->setTensorBuffer(convScaleBuffer, 1 * sizeof(float), syn_type_float);
    convertScale->setTensorBuffer(convScaleBuffer, 1 * sizeof(float), syn_type_float);

    synConvolutionParams params;
    params.dH   = weightsStride;
    params.dW   = weightsStride;
    params.kH   = S;
    params.kW   = R;
    params.padT = weightsPadding;
    params.padB = weightsPadding;
    params.padL = weightsPadding;

    params.padR = weightsPadding;
    params.dilH = 1;
    params.dilW = 1;

    NodePtr convCguid = NodeFactory::createNode({convIn, convWeight, nullptr, inScale, weightScale, nullptr},
                                                {convOut},
                                                &params,
                                                sizeof(synConvolutionParams),
                                                "conv2d_fp8_bf16",
                                                "conv_node");

    NodePtr convert = NodeFactory::createNode({convOut, convertScale},
                                              {convertOut},
                                              nullptr,
                                              "convert_to_fp8_bf16",
                                              "convert_node");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, convCguid));
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, convert));
    ASSERT_EQ(m_graph->getNodes().size(), 2);

    m_graph->setInferenceMode(true);

    ASSERT_TRUE(fuseConvertMme(*m_graph));

    // verify fusion has occurred and guid is now conv2d_fp8_hf8
    ASSERT_EQ(m_graph->getNodes().size(), 1);
    ASSERT_EQ(m_graph->getExeSortedNodes().front()->getGUID(), "conv2d_fp8_hf8");

    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
    // verify cguid extracted and now we have conv with fp8 output
    ASSERT_EQ(m_graph->getNodes().size(), 1);
    ASSERT_EQ(m_graph->getExeSortedNodes().front()->getGUID(), "spatial_convolution");
    MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(m_graph->getExeSortedNodes().front());
    ASSERT_TRUE(mmeNode != nullptr);
    const MmeExpBias& MmeExpBias = mmeNode->getMmeExpBias();
    ASSERT_EQ(MmeExpBias.fp8BiasIn[0], 11);
    ASSERT_EQ(MmeExpBias.fp8BiasIn[1], 11);
    ASSERT_EQ(MmeExpBias.fp8BiasOut, 3);
}

INSTANTIATE_TEST_SUITE_P(,
                         FuseConvertMmeTest,
                         ::testing::Values(synDeviceGaudi2, synDeviceGaudi3),
                         GenericGraphTest::GetName());