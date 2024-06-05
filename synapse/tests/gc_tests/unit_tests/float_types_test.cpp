#include <vector>

#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "syn_logging.h"
#include "tensor.h"
#include "test_utils.h"

class FloatTypesTestParameterized :
    public GraphOptimizerTest, public testing::WithParamInterface<std::tuple<float, float>> {};
class FloatTypesTestParameterizedBf16 : public FloatTypesTestParameterized {};
class FloatTypesTestParameterizedFp16 : public FloatTypesTestParameterized
{};
/*
 * Tests the pass staticTensorsFloatConversion by compiling a single conv node.
 * All weights tensor elements are same float value.
 * Pass should succeed and weights tensor elements be converted to expected bf16 value.
 */
TEST_P(FloatTypesTestParameterizedBf16, test_float_conversion_pass_bf16) {

    Gaudi2Graph      g;

    const TSize batch = 1;
    const TSize kW = 1;
    const TSize kH = 1;
    const TSize dW = 1;
    const TSize dH = 1;
    const TSize nOFM = 16;
    const TSize wOFM = 4;
    const TSize hOFM = 4;
    const TSize nIFM = 16;
    const TSize padH = 0;
    const TSize padW = 0;
    //o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * dW) + kW - (2 * padW);
    const TSize hIFM = ((hOFM - 1) * dH) + kH - (2 * padH);

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

    const TSize inTensorDims[]     = {nIFM, wIFM, hIFM, batch};
    const TSize weightTensorDims[] = {nOFM, nIFM, kW, kH};
    const TSize outTensorDims[]    =  {nOFM, wOFM, hOFM, batch};

    unsigned inputSize  = batch * nIFM * wIFM * hIFM ;
    unsigned weightSize = nIFM * nOFM * kW * kH ;

    std::vector<float> inData(inputSize);
    std::fill(inData.begin(), inData.end(), 1);
    const float origValue     = std::get<0>(GetParam());
    const float expectedValue = std::get<1>(GetParam());
    std::vector<float> weightData(weightSize);
    std::fill(weightData.begin(), weightData.end(), origValue); // all weight tensors elements are same


    float* ifm     = inData.data();
    float* weights = weightData.data();

    TensorPtr opATensor = TensorPtr(new Tensor(4U, inTensorDims, syn_type_bf16, reinterpret_cast<char*>(ifm)));
    TensorPtr opBTensor = TensorPtr(new Tensor(4U, weightTensorDims, syn_type_bf16, reinterpret_cast<char*>(weights)));
    TensorPtr outTensor = TensorPtr(new Tensor(4U, outTensorDims, syn_type_bf16));

    opBTensor->setAsWeights();
    opBTensor->setAsStaticParam();

    NodePtr n = getConvNodeWithGoyaLayouts(opATensor, opBTensor, nullptr, outTensor, params, "conv_node");
    GraphEditor::addNode(g, n);

    // Disable the weights compression so we can test the weights values
    setGlobalConfForTest(GCFG_ENABLED_COMPRESSION_MODES, "0");

    ASSERT_TRUE(staticTensorsFloatConversion(g));

    //Test that weight tensor data was converted from float to bf16 expected value
    TensorPtr w         = *g.getTensors().find(opBTensor);
    bfloat16* bf16Data  = reinterpret_cast<bfloat16*>(w->getData());
    unsigned  numErrors = 0;
    for (unsigned i = 0; i < weightSize; i++ )
    {
        if ((float)bf16Data[i] - expectedValue != 0.0)
        {
            numErrors++;
            LOG_ERR(SYN_TEST, "index ({}) - : actual value({}), expected value ({})", i, (float)bf16Data[i], expectedValue);
        }
    }
    ASSERT_EQ(numErrors, 0);
}

INSTANTIATE_TEST_SUITE_P(_,
                         FloatTypesTestParameterizedBf16,
                        testing::Values(std::make_tuple(5.046875, 5.0625), // weights tensor data rounded up
                                        std::make_tuple(5.0078125, 5.0))   // weights tensor data rounded down
);

TEST_P(FloatTypesTestParameterizedFp16, test_float_conversion_pass_fp16) {

    Gaudi2Graph      g;

    const TSize dimSize    = 9;
    const TSize opsDims [] = {dimSize, dimSize};
    const TSize inputSize  = dimSize * dimSize;
    const float origValue     = std::get<0>(GetParam());
    const float expectedValue = std::get<1>(GetParam());

    std::vector<float> opAData(inputSize);
    std::fill(opAData.begin(), opAData.end(), origValue); // all opA tensor elements are same

    std::vector<float> opBData(inputSize);
    std::fill(opBData.begin(), opBData.end(), 1);

    TensorPtr opATensor = TensorPtr(new Tensor(2U, opsDims, syn_type_fp16, reinterpret_cast<char*>(opAData.data())));
    TensorPtr opBTensor = TensorPtr(new Tensor(2U, opsDims, syn_type_fp16, reinterpret_cast<char*>(opBData.data())));
    TensorPtr outTensor = TensorPtr(new Tensor(2U, opsDims, syn_type_float));

    opATensor->setAsStaticParam();

    NodePtr n =  NodeFactory::createNode({opATensor, opBTensor}, {outTensor}, nullptr ,NodeFactory::gemmNodeTypeName ,"gemm");
    GraphEditor::addNode(g, n);

    ASSERT_TRUE(staticTensorsFloatConversion(g));

    //Test opA tensor data was converted from float to bf16 expected value
    TensorPtr A          = *g.getTensors().find(opATensor);
    fp16_t*   fp16Data   = reinterpret_cast<fp16_t*>(A->getData());
    unsigned numErrors = 0;
    for (unsigned i = 0; i < inputSize; i++)
    {
        if ((float)fp16Data[i] != expectedValue)
        {
            numErrors++;
            LOG_ERR(SYN_TEST, "index ({}) - : actual value({}), expected value ({})", i, (float)fp16Data[i], expectedValue);
        }
    }
    ASSERT_EQ(numErrors, 0);
}

INSTANTIATE_TEST_SUITE_P(
    _,
    FloatTypesTestParameterizedFp16,
    testing::Values(std::make_tuple(2.5, 2.5),                  // no rounding simple
                    std::make_tuple(2.501953125, 2.501953125),  // no rounding - mantissa 10th bit set
                    std::make_tuple(2.5009765625, 2.5),         // round down to nearest even
                    std::make_tuple(2.5029296875, 2.50390625),  // round up to nearest even
                    std::make_tuple(float(fp16_t::max()) + 1.0, float(fp16_t::max())),       // clip to max
                    std::make_tuple(float(fp16_t::lowest()) - 1.0, float(fp16_t::lowest()))  // clip to lowest negative
                    ));
