#include <vector>
#include "node_factory.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "test_utils.h"
#include "utils.h"
#include "scoped_configuration_change.h"

typedef std::tuple<synDataType, synDataType, synDataType> dataTypeTuple;   // for easy param get
typedef std::vector<synDataType>                          dataTypeVector;  // for easy compare against test result

class CastNonTpcNodesPassTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<bool, dataTypeTuple, dataTypeVector>>
{
protected:
    void init();

    void runTest(HabanaGraph& g);

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
    std::vector<float> inputData   = {};
    std::vector<float> weightsData = {};
    std::vector<float> biasData    = {};

    synConvolutionParams params;
};

void CastNonTpcNodesPassTest::init()
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

void CastNonTpcNodesPassTest::runTest(HabanaGraph& g)
{
    const bool should_cast                       = std::get<0>(GetParam());
    std::tie(inputType, weightsType, outputType) = std::get<1>(GetParam());
    const dataTypeVector& expectedDataType       = std::get<2>(GetParam());

    TensorPtr inputTensor =
        TensorPtr(new Tensor(4U, inputSizes.data(), inputType, reinterpret_cast<char*>(inputData.data())));

    TensorPtr weightsTensor =
        TensorPtr(new Tensor(4U, weightsSizes.data(), weightsType, reinterpret_cast<char*>(weightsData.data())));
    weightsTensor->setAsStaticParam(true);
    weightsTensor->setAsWeights();

    TensorPtr outputTensor = TensorPtr(new Tensor(4U, outSizes.data(), outputType));

    NodePtr conv = NodeFactory::createNode({inputTensor, weightsTensor},
                                           {outputTensor},
                                           &params,
                                           NodeFactory::convolutionNodeTypeName,
                                           "conv_node");

    GraphEditor::addNode(g, conv);
    if (should_cast)
    {
        ASSERT_TRUE(castForNonTPCNodes(g));
    }
    else
    {
        ASSERT_FALSE(castForNonTPCNodes(g));
    }

    // compare casted types to expected
    dataTypeVector dataTypeAfterCast = std::vector<synDataType>();

    TensorVector castedInputs  = conv->getInputs();
    TensorVector castedOutputs = conv->getOutputs();
    for (TensorPtr tensor : castedInputs)
    {
        dataTypeAfterCast.push_back(tensor->getElementType());
    }
    dataTypeAfterCast.push_back(castedOutputs[0]->getElementType());

    ASSERT_EQ(dataTypeAfterCast, expectedDataType);
}
class CastNonTpcNodesPassGaudi2 : public CastNonTpcNodesPassTest
{
};

TEST_P(CastNonTpcNodesPassGaudi2, cast_non_tpc_test_gaudi2)
{
    //op validation disabled since tests uses unmatched tensors dtypes
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    init();

    Gaudi2Graph g;
    g.setInferenceMode(true);

    runTest(g);
}

INSTANTIATE_TEST_SUITE_P(
    ,  //                               input      | weights      | output
    CastNonTpcNodesPassGaudi2,
    testing::Values(std::make_tuple(  // cast to defaults when needed
                        true,
                        dataTypeTuple(syn_type_int8, syn_type_int32, syn_type_float),
                        dataTypeVector {syn_type_bf16, syn_type_bf16, syn_type_float}),
                    std::make_tuple(  // don't cast
                        true,
                        dataTypeTuple(syn_type_bf16, syn_type_bf16, syn_type_float),
                        dataTypeVector {syn_type_bf16, syn_type_bf16, syn_type_float}),
                    std::make_tuple(  // cast input as weights - bfloat
                        true,
                        dataTypeTuple(syn_type_fp16, syn_type_bf16, syn_type_float),
                        dataTypeVector {syn_type_bf16, syn_type_bf16, syn_type_float}),
                    std::make_tuple(  // cast all to default
                        true,
                        dataTypeTuple(syn_type_uint8, syn_type_int8, syn_type_int8),
                        dataTypeVector {syn_type_bf16, syn_type_bf16, syn_type_bf16}),
                    std::make_tuple(  // cast all to default
                        true,
                        dataTypeTuple(syn_type_int16, syn_type_int8, syn_type_int32),
                        dataTypeVector {syn_type_bf16, syn_type_bf16, syn_type_bf16}),
                    std::make_tuple(  // cast input as weights
                        true,
                        dataTypeTuple(syn_type_int32, syn_type_float, syn_type_float),
                        dataTypeVector {syn_type_float, syn_type_float, syn_type_float}),
                    std::make_tuple(  // cast cast_hf8_to_f8 does not exist, don't cast and return false.
                        false,
                        dataTypeTuple(syn_type_fp8_143, syn_type_fp8_152, syn_type_int32),
                        dataTypeVector {syn_type_fp8_143, syn_type_fp8_152, syn_type_int32}),
                    std::make_tuple(  // cast output to default
                        true,
                        dataTypeTuple(syn_type_fp8_152, syn_type_fp8_152, syn_type_int32),
                        dataTypeVector {syn_type_fp8_152, syn_type_fp8_152, syn_type_bf16}),
                    std::make_tuple(  // cast input as weights
                        true,
                        dataTypeTuple(syn_type_fp8_152, syn_type_bf16, syn_type_bf16),
                        dataTypeVector {syn_type_bf16, syn_type_bf16, syn_type_bf16}),
                    std::make_tuple(  // cast input as weights
                        true,
                        dataTypeTuple(syn_type_bf16, syn_type_fp8_152, syn_type_fp8_152),
                        dataTypeVector {syn_type_fp8_152, syn_type_fp8_152, syn_type_fp8_152}),
                    std::make_tuple(  // cast input as weights
                        true,
                        dataTypeTuple(syn_type_fp8_143, syn_type_bf16, syn_type_bf16),
                        dataTypeVector {syn_type_bf16, syn_type_bf16, syn_type_bf16}),
                    std::make_tuple(  // cast input as weights
                        true,
                        dataTypeTuple(syn_type_bf16, syn_type_fp8_143, syn_type_fp8_143),
                        dataTypeVector {syn_type_fp8_143, syn_type_fp8_143, syn_type_fp8_143})));