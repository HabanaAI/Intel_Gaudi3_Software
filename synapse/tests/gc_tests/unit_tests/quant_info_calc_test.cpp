#include <math.h>
#include "quantization_data.h"
#include "sim_graph.h"
#include "syn_singleton.hpp"
#include "synapse_common_types.h"
#include "tensor.h"
#include "node.h"
#include "sim_graph.h"
#include "infra/global_conf_manager.h"
#include "graph_optimizer_test.h"
#include "test_utils.h"
#include <graph_compiler/passes/quantization_utils.h>
#include <graph_compiler/habana_nodes/node_factory.h>
#include "gaudi2_graph.h"

// used for section handle objects in gc tests level
extern ConcurrentSlotMapAlloc<InternalSectionHandle> sectionHndlSlopMap;

using namespace gc;

// TODO SW-136092 - Enable and update after moving quantization passes to Gaudi2
class QuantInfoCalcTest : public GraphOptimizerTest
{
protected:
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_CALC_DYNAMIC_RANGE, "true");
    }
};

TEST_F(QuantInfoCalcTest, DISABLED_conv_calc_quant)
{
    Gaudi2Graph g;

    bool ret;
    int IFMMin = -2;
    int IFMMax = 2;
    int WMin = -2;
    int WMax = 2;
    int OFMMin = -4;
    int OFMMax = 4;

    TSize batch = 1;

    const TSize kW   = 3;
    const TSize kH   = 3;
    const TSize dW   = 1;
    const TSize dH   = 1;
    const TSize nOFM = 129;
    const TSize wOFM = 165;
    const TSize hOFM = 19;
    const TSize nIFM = 64;
    const TSize padH = 0;
    const TSize padW = 0;

    synConvolutionParams params;
    params.dH   = dH;
    params.dW   = dW;
    params.kH   = kH;
    params.kW   = kW;
    params.padT = padH;
    params.padB = padH;
    params.padL = padW;
    params.padR = padW;
    params.dilH = 1;
    params.dilW = 1;

    //o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * params.dW) + (params.kW - 1) * params.dilW + 1 - (params.padL + params.padR);
    const TSize hIFM = ((hOFM - 1) * params.dH) + (params.kH - 1) * params.dilH + 1 - (params.padT + params.padB);

    char* ifm      = new char[nIFM * wIFM * hIFM * batch];
    float* weights = new float[nIFM * nOFM * params.kW * params.kH];

    std::generate (ifm, ifm + nIFM * wIFM * hIFM * batch, Test_Random_Number_Creator (std::array<int, 2>({IFMMin, IFMMax})));
    std::generate (weights, weights + nIFM * nOFM * params.kW * params.kH, Test_Random_Number_Creator (std::array<int, 2>({WMin, WMax})));

    const TSize i_sizes[] = { nIFM, wIFM, hIFM, batch };
    const TSize o_sizes[] = { nOFM, wOFM, hOFM, batch };
    const TSize w_sizes[] = { nOFM, nIFM, params.kW, params.kH };

    TensorPtr IFM     = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed, reinterpret_cast<char*>(ifm)));
    TensorPtr OFM_ref = TensorPtr(new Tensor(4U, o_sizes, syn_type_fixed));
    TensorPtr OFM     = TensorPtr(new Tensor(4U, o_sizes, syn_type_fixed));
    TensorPtr W       = TensorPtr(new Tensor(4U, w_sizes, syn_type_fixed, reinterpret_cast<char*>(weights)));

    DynamicRange ifmDR;
    ifmDR.min = IFMMin;
    ifmDR.max = IFMMax;
    ifmDR.isSet = true;
    IFM->setDynamicRange(ifmDR);
    IFM->setName("IFM");

    DynamicRange wDR;
    wDR.min = WMin;
    wDR.max = WMax;
    wDR.isSet = true;
    W->setDynamicRange(wDR);
    W->setAsStaticParam();
    W->setName("WEIGHTS");

    DynamicRange ofmDR;
    ofmDR.min = OFMMin;
    ofmDR.max = OFMMax;
    ofmDR.isSet = true;
    OFM->setDynamicRange(ofmDR);
    OFM->setName("OFM");

    //Validate the output first (as tensors may be modified by Graph later on)
    SimGraph ref_g;

    NodePtr n = getConvNodeWithGoyaLayouts(IFM, W, nullptr, OFM_ref, params, "");
    GraphEditor::addNode(ref_g, n);

    ret = ref_g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    ret = ref_g.execute();
    ASSERT_EQ(ret, true) << "Failed to execute graph";

    n = getConvNodeWithGoyaLayouts(IFM, W, nullptr, OFM, params, "");

    NodeAnnotation& ann = n->getNodeAnnotation();
    ann.mmeMetaData.packing[0] = 1;

    GraphEditor::addNode(g, n);

    ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    ret = g.execute();
    ASSERT_EQ(ret, true) << "Failed to execute graph";

    double zpWRef, scaleWRef, zpIFMRef, scaleIFMRef, zpOFMRef, scaleOFMRef;
    QuantizationUtils::calcDataScaleZp(IFMMin, IFMMax, QuantizationData::synTypeToQuantType(syn_type_fixed), scaleIFMRef, zpIFMRef, false);
    QuantizationUtils::calcDataScaleZp(WMin, WMax, QuantizationData::synTypeToQuantType(syn_type_fixed), scaleWRef, zpWRef, false);
    QuantizationUtils::calcDataScaleZp(OFMMin, OFMMax, QuantizationData::synTypeToQuantType(syn_type_fixed), scaleOFMRef, zpOFMRef, false);

    double scaleIFM = IFM->getQuantizationParams().scale();
    double zpIFM = IFM->getQuantizationParams().zp();
    double scaleW = W->getQuantizationParams().scale();
    double zpW = W->getQuantizationParams().zp();
    double scaleOFM = OFM->getQuantizationParams().scale();
    double zpOFM = OFM->getQuantizationParams().zp();

    ASSERT_EQ(scaleIFM, scaleIFMRef) << "Wrong scale used for IFM tensor: " << scaleIFM << " Ref: " << scaleIFMRef;
    ASSERT_EQ(zpIFM, zpIFMRef) << "Wrong Zero Point used for IFM tensor: " << zpIFM << " Ref: " << zpIFMRef;
    ASSERT_EQ(scaleW, scaleWRef) << "Wrong scale used for Weight tensor: " << scaleW << " Ref: " << scaleWRef;
    ASSERT_EQ(zpW, zpWRef) << "Wrong Zero Point used for Weight tensor: " << zpW << " Ref: " << zpWRef;
    ASSERT_EQ(scaleOFM, scaleOFMRef) << "Wrong scale used for OFM tensor: " << scaleOFM << " Ref: " << scaleOFMRef;
    ASSERT_EQ(zpOFM, zpOFMRef) << "Wrong Zero Point used for OFM tensor: " << zpOFM << " Ref: " << zpOFMRef;

    delete[] weights;
    delete[] ifm;
}

TEST_F(QuantInfoCalcTest, DISABLED_multiple_tpc_nodes_quant)
{
    Gaudi2Graph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;
    float in1[n * w * h * batch] = {1, 3, 5, 7, 2, 4, 6, 8, 10};
    float in2[n * w * h * batch] = {1, 3, 0, 1, 2, 4, 2, 3, 1};
    float in3[n * w * h * batch] = {1, 2, 2, 1, 2, 1, 2, 0, -1};
    float in4[n * w * h * batch] = {1, 8, -2, 1, 4, -1, 4, 7, 0};
    const TSize sizes[] = {n, w, h, batch};

    // Static Tensor with Dynamic range set
    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    t1->setName("t1");
    DynamicRange t1DR;
    t1DR.min = 0;
    t1DR.max = 10;
    t1DR.isSet = true;
    t1->setDynamicRange(t1DR);
    t1->setAsStaticParam();

    // Dynamic Tensor with Dynamic range set
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in2)));
    t2->setName("t2");
    DynamicRange t2DR;
    t2DR.min = 0;
    t2DR.max = 4;
    t2DR.isSet = true;
    t2->setDynamicRange(t2DR);

    // Static Tensor without setting Dynamic range
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in3)));
    t3->setName("t3");
    t3->setAsStaticParam();

    // Dynamic Tensor without setting Dynamic range
    pTensor t4 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in4)));
    t4->setName("t4");

    pNode tanh = NodeFactory::createNode({t1}, {t2}, nullptr, "tanh_i16", "tanh");
    GraphEditor::addNode(g, tanh);
    pNode abs = NodeFactory::createNode({t2}, {t3}, nullptr, "abs_i16", "abs");
    GraphEditor::addNode(g, abs);
    pNode sin = NodeFactory::createNode({t3}, {t4}, nullptr, "sin_i16", "sin");
    GraphEditor::addNode(g, sin);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    double t1zpRef, t1scaleRef, t2zpRef, t2scaleRef, t3zpRef, t3scaleRef, t4zpRef = 0, t4scaleRef = 1;
    QuantizationUtils::calcDataScaleZp(0, 10, quant_type_int16, t1scaleRef, t1zpRef, false);
    QuantizationUtils::calcDataScaleZp(0, 4, quant_type_int16, t2scaleRef, t2zpRef, false);
    QuantizationUtils::calcDataScaleZp(-2, 2, quant_type_int16, t3scaleRef, t3zpRef, false);

    double t1scale = t1->getQuantizationParams().scale();
    double t1zp = t1->getQuantizationParams().zp();
    double t2scale = t2->getQuantizationParams().scale();
    double t2zp = t2->getQuantizationParams().zp();
    double t3scale = t3->getQuantizationParams().scale();
    double t3zp = t3->getQuantizationParams().zp();
    double t4scale = t4->getQuantizationParams().scale();
    double t4zp = t4->getQuantizationParams().zp();

    ASSERT_EQ(t1scale, t1scaleRef) << "Wrong scale used for t1 tensor: " << t1scale << " Ref: " << t1scaleRef;
    ASSERT_EQ(t1zp, t1zpRef) << "Wrong Zero Point used for t1 tensor: " << t1zp << " Ref: " << t1zpRef;
    ASSERT_EQ(t2scale, t2scaleRef) << "Wrong scale used for t2 tensor: " << t2scale << " Ref: " << t2scaleRef;
    ASSERT_EQ(t2zp, t2zpRef) << "Wrong Zero Point used for t2 tensor: " << t2zp << " Ref: " << t2zpRef;
    ASSERT_EQ(t3scale, t3scaleRef) << "Wrong scale used for t3 tensor: " << t3scale << " Ref: " << t3scaleRef;
    ASSERT_EQ(t3zp, t3zpRef) << "Wrong Zero Point used for t3 tensor: " << t3zp << " Ref: " << t3zpRef;
    ASSERT_EQ(t4scale, t4scaleRef) << "Wrong scale used for t4 tensor: " << t4scale << " Ref: " << t4scaleRef;
    ASSERT_EQ(t4zp, t4zpRef) << "Wrong Zero Point used for t4 tensor: " << t4zp << " Ref: " << t4zpRef;
}

TEST_F(QuantInfoCalcTest, DISABLED_enforce_node_precision)
{
    // Enable enforceNodePrecision Pass
    GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(true);
    Gaudi2Graph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;
    float in1[n * w * h * batch] = {1, 3, 5, 7, 2, 4, 6, 8, 10};
    float in2[n * w * h * batch] = {1, 3, 0, 1, 2, 4, 2, 3, 1};
    float in3[n * w * h * batch] = {1, 2, 2, 1, 2, 1, 2, 0, -1};
    float in4[n * w * h * batch] = {1, 8, -2, 1, 4, -1, 4, 7, 0};
    float in5[n * w * h * batch] = {1, 4, -3, 1, 6, -1, 8, 7, 8};
    float in6[n * w * h * batch] = {3, 8, -2, 1, 4, 5, -4, 4, 0};
    const TSize sizes[] = {n, w, h, batch};

    // Static Tensor with Dynamic range set
    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in1)));
    t1->setName("t1");
    DynamicRange t1DR;
    t1DR.min = 0;
    t1DR.max = 10;
    t1DR.isSet = true;
    t1->setDynamicRange(t1DR);
    t1->setAsStaticParam();

    // Dynamic Tensor with Dynamic range set
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in2)));
    t2->setName("t2");
    DynamicRange t2DR;
    t2DR.min = 0;
    t2DR.max = 4;
    t2DR.isSet = true;
    t2->setDynamicRange(t2DR);

    // Static Tensor without setting Dynamic range
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in3)));
    t3->setName("t3");
    t3->setAsStaticParam();

    // Dynamic Tensor without setting Dynamic range
    pTensor t4 = pTensor(new Tensor(4U, sizes, syn_type_int16, reinterpret_cast<char*>(in4)));
    t4->setName("t4");

    // Static Tensor with user quantization
    pTensor t5 = pTensor(new Tensor(4U, sizes, syn_type_int8, reinterpret_cast<char*>(in5)));
    t5->setName("t5");
    synQuantizationParams userQuantParams1(syn_type_int8);
    userQuantParams1.m_scale = 0;
    userQuantParams1.m_zp = 1;
    QuantizationData userQuantData1(userQuantParams1);
    t5->setQuantizationParams(userQuantData1);
    t5->setAsStaticParam();

    // Dynamic Tensor with user quantization
    pTensor t6 = pTensor(new Tensor(4U, sizes, syn_type_int8, reinterpret_cast<char*>(in6)));
    t6->setName("t6");
    synQuantizationParams userQuantParams2(syn_type_int8);
    userQuantParams2.m_scale = 0;
    userQuantParams2.m_zp = 1;
    QuantizationData userQuantData2(userQuantParams2);
    t6->setQuantizationParams(userQuantData2);

    pNode tanh = NodeFactory::createGenericTPCNode({t1}, {t2}, nullptr, "tanh_i16", "tanh");
    GraphEditor::addNode(g, tanh);
    pNode abs = NodeFactory::createGenericTPCNode({t2}, {t3}, nullptr, "abs_i8", "abs");
    GraphEditor::addNode(g, abs);
    pNode sin = NodeFactory::createGenericTPCNode({t3}, {t4}, nullptr, "sin_i8", "sin");
    GraphEditor::addNode(g, sin);
    pNode cos = NodeFactory::createGenericTPCNode({t4}, {t5}, nullptr, "cos_i8", "cos");
    GraphEditor::addNode(g, cos);
    pNode tan = NodeFactory::createGenericTPCNode({t5}, {t6}, nullptr, "tan_i8", "tan");
    GraphEditor::addNode(g, tan);

    ASSERT_TRUE(enforceNodePrecision(g)); // run pass
    ASSERT_EQ(t1->getElementType(), syn_type_int16) << "Wrong Element type found for t1 tensor: " << t1->getElementType();
    ASSERT_EQ(t2->getElementType(), syn_type_int16) << "Wrong Element type found for t2 tensor: " << t2->getElementType();
    ASSERT_EQ(t3->getElementType(), syn_type_single) << "Wrong Element type found for t3 tensor: " << t3->getElementType();
    ASSERT_EQ(t4->getElementType(), syn_type_single) << "Wrong Element type found for t4 tensor: " << t4->getElementType();
    ASSERT_EQ(t5->getElementType(), syn_type_int8) << "Wrong Element type found for t3 tensor: " << t5->getElementType();
    ASSERT_EQ(t6->getElementType(), syn_type_int8) << "Wrong Element type found for t4 tensor: " << t6->getElementType();
}

TEST_F(QuantInfoCalcTest, DISABLED_non_float_buffer_data_type)
{
    Gaudi2Graph g;

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;
    const TSize numElements = n * w * h * batch;
    int16_t in1[numElements] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    const TSize sizes[] = {n, w, h, batch};

    // Static Tensor with Data buffer set
    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_na));
    t1->setName("t1");
    t1->setAsStaticParam();
    t1->setTensorBuffer(in1, numElements * sizeof(int16_t), syn_type_int16);

    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_int16));
    t2->setName("t2");

    pNode sinNode = NodeFactory::createNode({t1}, {t2}, nullptr, "sin_f32", "sinNode");
    GraphEditor::addNode(g, sinNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    double t1zpRef, t1scaleRef;
    QuantizationUtils::calcDataScaleZp(1, 9, quant_type_int16, t1scaleRef, t1zpRef, false);

    double t1scale = t1->getQuantizationParams(syn_type_int16).scale();
    double t1zp = t1->getQuantizationParams(syn_type_int16).zp();

    ASSERT_EQ(t1scale, t1scaleRef) << "Wrong scale used for t1 tensor: " << t1scale << " Ref: " << t1scaleRef;
    ASSERT_EQ(t1zp, t1zpRef) << "Wrong Zero Point used for t1 tensor: " << t1zp << " Ref: " << t1zpRef;
}

TEST_F(QuantInfoCalcTest, test_calc_expBias_fp8_152)
{
    double value = 0.8;
    unsigned referenceExpBias = QuantizationData::S_EXP_BIAS_152_DEFAULT;
    unsigned calculatedExpBias = 0;
    for (unsigned i = 0; i < 32; i++)
    {
        ASSERT_TRUE(QuantizationUtils::calcExpBias(value, quant_type_fp8_152, calculatedExpBias));
        EXPECT_EQ(calculatedExpBias, referenceExpBias);
        // prepare next iteration
        value *= 2;
    }
}

TEST_F(QuantInfoCalcTest, test_calc_expBias_fp8_143)
{
    double value = 0.8;
    unsigned referenceExpBias = 15;
    unsigned calculatedExpBias = 0;
    ASSERT_TRUE(QuantizationUtils::calcExpBias(value, quant_type_fp8_143, calculatedExpBias));
    EXPECT_EQ(calculatedExpBias, referenceExpBias);

    value = 12;
    referenceExpBias = 11;
    ASSERT_TRUE(QuantizationUtils::calcExpBias(value, quant_type_fp8_143, calculatedExpBias));
    EXPECT_EQ(calculatedExpBias, referenceExpBias);

    value = 201;
    referenceExpBias = 7;
    ASSERT_TRUE(QuantizationUtils::calcExpBias(value, quant_type_fp8_143, calculatedExpBias));
    EXPECT_EQ(calculatedExpBias, referenceExpBias);

    value = 2798;
    referenceExpBias = 3;
    ASSERT_TRUE(QuantizationUtils::calcExpBias(value, quant_type_fp8_143, calculatedExpBias));
    EXPECT_EQ(calculatedExpBias, referenceExpBias);

    // test if value is overflowed (in this case scale will be set)
    value = 3841;
    referenceExpBias = 3;
    ASSERT_TRUE(QuantizationUtils::calcExpBias(value, quant_type_fp8_143, calculatedExpBias));
    EXPECT_EQ(calculatedExpBias, referenceExpBias);
}

TEST_F(QuantInfoCalcTest, test_compile_with_fp8)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setQuantizationEnabled(true);

    const TSize H = 5;
    const TSize W = 5;
    const TSize C = 2;
    const TSize N = 1;
    const TSize K = 2;
    const TSize R = 2;
    const TSize S = 2;

    const std::vector<TSize> inputSizes = {C, W, H, N};

    TensorPtr    addIn1  = TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_bf16));
    TensorPtr    addIn2  = TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_bf16));
    TensorPtr    castIn  = TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_bf16));


    const TSize weightsStride  = 1;
    const TSize weightsPadding = 1;

    const std::vector<TSize> weightsSizes = {K, C, S, R};

    const TSize outW = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outH = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
    const TSize outC = K;
    //                                     C    W    H    N
    const std::vector<TSize> outSizes = {outC, outW, outH, N};

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

    synDataType bufferType = syn_type_bf16;

    auto weightsNumElem     = std::accumulate(weightsSizes.begin(), weightsSizes.end(), 1U, std::multiplies<TSize>());
    bfloat16* weightBuffer  = reinterpret_cast<bfloat16*>(allocateBufferForSynType(bufferType, weightsNumElem));
    float weightMaxPositive = 240;
    // set the weights data range (-weightMaxPositive - 1, weightMaxPositive), this way we can test that we take
    // the maxAbs in exp bias calculation: maxAbs(-weightMaxPositive - 1) * (backoff-factor, defualt is 1.0) = 241 corresponds to expBias 3
    fillWithRandom(weightBuffer,
                   weightsNumElem,
                   std::pair<float, float>(-weightMaxPositive -1 , weightMaxPositive));
    TensorPtr weightInput = TensorPtr(new Tensor(weightsSizes.size(), weightsSizes.data(), syn_type_fp8_143));
    weightInput->setTensorBuffer(reinterpret_cast<char*>(weightBuffer),
                                 weightsNumElem * dataTypeSizeInBytes(bufferType),
                                 bufferType);
    weightInput->setAsStaticParam(true);

    TensorPtr    convIn  = TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_fp8_143));
    TensorPtr    convOut = TensorPtr(new Tensor(4U, outSizes.data(), syn_type_bf16));

    // set dynamic ranges
    constexpr float min = 0.0;
    DynamicRange    add1InputDR;
    add1InputDR.min   = min;
    add1InputDR.max   = 0.9374; // corresponds to exp bias 15
    add1InputDR.isSet = true;
    addIn1->setDynamicRange(add1InputDR);
    DynamicRange add2InputDR;
    add2InputDR.min   = min;
    add2InputDR.max   = 15; // corresponds to exp bias 11
    add2InputDR.isSet = true;
    addIn2->setDynamicRange(add2InputDR);
    DynamicRange convInputDR;
    convInputDR.min   = min;
    convInputDR.max   = 240; // corresponds to exp bias 7
    convInputDR.isSet = true;
    convIn->setDynamicRange(convInputDR);
    castIn->setDynamicRange(convInputDR);
    DynamicRange outputDR;
    outputDR.min   = min;
    outputDR.max   = 3840; // corresponds to exp bias 3
    outputDR.isSet = true;
    convOut->setDynamicRange(outputDR);

    uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2;
    synMemoryDescriptor persistentMemoryDesc(true);

    addIn1->setDramOffset(0x7000);
    addIn1->setMemorySectionID(memSecId);
    addIn1->setMemoryDescriptor(persistentMemoryDesc);
    addIn2->setDramOffset(0x8000);
    addIn2->setMemorySectionID(memSecId);
    addIn2->setMemoryDescriptor(persistentMemoryDesc);
    convOut->setDramOffset(0x9000);
    convOut->setMemorySectionID(memSecId);
    convOut->setMemoryDescriptor(persistentMemoryDesc);
    // set the weights in section id
    auto sectionHandle = sectionHndlSlopMap.insert(0, 0);
    uint32_t sectionId               = g.getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);
    sectionHandle.second->setIDAndLock(sectionId);
    weightInput->setMemoryDescriptor(persistentMemoryDesc);
    weightInput->setSectionHandle(sectionHandle.second);
    weightInput->setMemorySectionID(sectionId);

    NodePtr addNode  = NodeFactory::createNode({addIn1, addIn2}, {castIn}, nullptr, "add_fwd_bf16", "add_node");
    NodePtr castNode = NodeFactory::createNode({castIn}, {convIn}, nullptr, "cast_bf16_to_hf8", "cast_node");
    NodePtr convNode = NodeFactory::createNode({convIn, weightInput},
                                               {convOut},
                                               &params,
                                               NodeFactory::convolutionNodeTypeName,
                                               "conv_node");

    ASSERT_TRUE(GraphEditor::addNode(g, addNode));
    ASSERT_TRUE(GraphEditor::addNode(g, castNode));
    ASSERT_TRUE(GraphEditor::addNode(g, convNode));

    ASSERT_TRUE(g.compile());
    // verify expBias is as expected according to the dynamic range max abs
    EXPECT_EQ(addIn1->getExpBias(syn_type_fp8_143), 15);
    EXPECT_EQ(addIn2->getExpBias(syn_type_fp8_143), 11);
    EXPECT_EQ(castIn->getExpBias(syn_type_fp8_143), 7);
    EXPECT_EQ(convIn->getExpBias(syn_type_fp8_143), 7);
    EXPECT_EQ(weightInput->getExpBias(syn_type_fp8_143), 3);
    EXPECT_EQ(convOut->getExpBias(syn_type_fp8_143), 3);
    EXPECT_EQ(addIn1->getExpBias(syn_type_fp8_152), QuantizationData::S_EXP_BIAS_152_DEFAULT);
    EXPECT_EQ(addIn2->getExpBias(syn_type_fp8_152), QuantizationData::S_EXP_BIAS_152_DEFAULT);
    EXPECT_EQ(convIn->getExpBias(syn_type_fp8_152), QuantizationData::S_EXP_BIAS_152_DEFAULT);
    EXPECT_EQ(weightInput->getExpBias(syn_type_fp8_152), QuantizationData::S_EXP_BIAS_152_DEFAULT);
    EXPECT_EQ(convOut->getExpBias(syn_type_fp8_152), QuantizationData::S_EXP_BIAS_152_DEFAULT);

    // verify conv node is still valid
    ASSERT_TRUE(convNode->validateNode());
}

#if 0
// consider porting to greco
TEST_F(QuantInfoCalcTest, conv_with_sparsity)
{
    bool ret;

    const TSize kW = 5;
    const TSize kH = 5;
    const TSize dW = 1;
    const TSize dH = 1;
    const TSize batch = 3;
    const TSize nOFM = 1;
    const TSize wOFM = 5;
    const TSize hOFM = 5;
    const TSize nIFM = 1;
    //o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW;
    const TSize hIFM = ((hOFM - 1) * dH) + kH;

    synConvolutionParams params;
    params.dH = dH;
    params.dW = dW;
    params.kH = kH;
    params.kW = kW;
    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;
    params.dilH = 1;
    params.dilW = 1;

    float* ifm = new float[batch * nIFM * wIFM * hIFM];
    float* weights = new float[batch * nIFM * nOFM * kW * kH];

    const float ifm_val = 2.f;
    for (unsigned i = 0; i < wIFM * hIFM * nIFM * batch; ++i)
    {
        ifm[i] = ifm_val;
    }
    for (unsigned i = 0; i < kW * kH * nIFM * nOFM; ++i)
    {
        weights[i] = i + 1;
    }

    const TSize i_sizes[] = { nIFM, wIFM, hIFM, batch };
    const TSize o_sizes[] = { nOFM, wOFM, hOFM, batch };
    const TSize w_sizes[] = { nOFM, nIFM, kW, kH };

    TensorPtr IFM1 = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed, reinterpret_cast<char*>(ifm)));
    TensorPtr OFM1 = TensorPtr(new Tensor(4U, o_sizes, syn_type_fixed));
    TensorPtr W1   = TensorPtr(new Tensor(4U, w_sizes, syn_type_fixed, reinterpret_cast<char*>(weights)));

    TensorPtr IFM2 = IFM1->clone();
    TensorPtr OFM2 = OFM1->clone();
    TensorPtr W2 = W1->clone();

    DynamicRange weightDR;
    weightDR.min = 1;
    weightDR.max = 10;
    weightDR.isSet = true;
    W1->setDynamicRange(weightDR);
    W1->setAsStaticParam();

    Gaudi2Graph g1;
    NodePtr  n1 = getConvNodeWithGoyaLayouts(IFM1, W1, nullptr, OFM1, params, "");
    g1.addNode(n1);
    ret = g1.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";
    ASSERT_EQ(W1->getAllQuantizationParams()[quant_type_int8].zp(), -128);

    // Same Graph with setting weights as sparsity, this should force zp=0
    W2->setDynamicRange(weightDR);
    W2->setAsStaticParam();
    W2->setAsSparsityWeights();

    Gaudi2Graph g2;
    NodePtr  n2 = getConvNodeWithGoyaLayouts(IFM2, W2, nullptr, OFM2, params, "");
    g2.addNode(n2);
    ret = g2.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";
    ASSERT_EQ(W2->getAllQuantizationParams()[quant_type_int8].zp(), 0);
}
#endif
