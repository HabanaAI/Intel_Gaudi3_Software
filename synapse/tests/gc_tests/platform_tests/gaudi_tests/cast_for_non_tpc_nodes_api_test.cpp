#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include <algorithm>
#include <cstddef>
#include <data_types/bfloat16.h>
#include <iterator>
#include <numeric>
#include "utils.h"
#include "data_type_utils.h"

class SynCastForNonTpcTest : public SynTrainingTestInfra
{
public:
    void castNonTpcApiTest(synDataType xTensorDtype, synDataType wTensorDtype, synDataType yTensorDtype);
    void setQuantData(void* data, unsigned tensorIndex, unsigned sizes[]);
};

void SynCastForNonTpcTest::castNonTpcApiTest(synDataType xTensorDtype,
                                             synDataType wTensorDtype,
                                             synDataType yTensorDtype)
{
    // Tested Graph:      ____
    //              x->->|    |
    //                   |Conv|->->y
    //              w->->|____|
    setGraphInferenceMode();
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

    const unsigned batch = 1;
    const unsigned nIFM  = 1;
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    const unsigned ofmDataSize = nOFM * wOFM * hOFM * batch;
    float          ofmRefBuffer[ofmDataSize];

    // create_tensor's layout
    unsigned dims          = 4;
    unsigned ifmDimSizes[] = {nIFM, wIFM, hIFM, batch};
    unsigned wghDimSizes[] = {nOFM, nIFM, params.kW, params.kH};
    unsigned ofmDimSizes[] = {nOFM, wOFM, hOFM, batch};

    unsigned xTensorIndex =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, ifmDimSizes, dims, syn_type_float);
    unsigned wTensorIndex = createConstPersistTensor(INPUT_TENSOR,
                                                     MEM_INIT_RANDOM_POSITIVE,
                                                     nullptr,
                                                     wghDimSizes,
                                                     dims,
                                                     syn_type_float,
                                                     nullptr,
                                                     "constWeightTensor");
    unsigned yTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_float);

    unsigned xTensorIndexCast =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ifmDimSizes, dims, xTensorDtype);
    unsigned wTensorIndexCast =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wghDimSizes, dims, wTensorDtype);
    unsigned yTensorIndexCast =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, yTensorDtype);

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];

    synDataType compareDtype = xTensorDtype;
    if (is8BitFloat(xTensorDtype))
    {
        setQuantData(xData, xTensorIndexCast, ifmDimSizes);
    }
    if (is8BitFloat(wTensorDtype))
    {
        // input should cast as weights
        compareDtype = wTensorDtype;
        setQuantData(wData, wTensorIndexCast, wghDimSizes);
    }

    // Adding cast node only for CPU conv calculations purposes
    addNodeToGraph(getCastGUID(syn_type_float, xTensorDtype).c_str(), {xTensorIndex}, {xTensorIndexCast});
    addNodeToGraph(getCastGUID(syn_type_float, wTensorDtype).c_str(), {wTensorIndex}, {wTensorIndexCast});
    addNodeToGraph(getCastGUID(yTensorDtype, syn_type_float).c_str(), {yTensorIndexCast}, {yTensorIndex});

    TensorIndices inputIndices  = {xTensorIndexCast, wTensorIndexCast};
    TensorIndices outputIndices = {yTensorIndexCast};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params,
                   sizeof(synConvolutionParams));

    compileTopology();
    runTopology();

    // Calculate reference on CPU using HPU casted weight
    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    calculateFwdConvolution(xDesc, (char*)xData, wDesc, (char*)wData, yDesc, (char*)ofmRefBuffer, params, m_deviceType);

    float*      pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    std::string errMsg;
    ASSERT_TRUE(
        compareFP8Results(ofmRefBuffer, pOutputBuffer, getNumberOfElements(ofmDimSizes), compareDtype, errMsg))
        << errMsg;
}

void SynCastForNonTpcTest::setQuantData(void* data, unsigned tensorIndex, unsigned sizes[])
{
    float val    = ((float*)data)[0];
    float maxVal = val;
    for (unsigned i = 0; i < getNumberOfElements(sizes); i++)
    {
        val    = abs(((float*)data)[i]);
        maxVal = maxVal > val ? maxVal : val;
    }

    synQuantDynamicRange xDynamicRange;
    xDynamicRange.max = maxVal + 1;
    xDynamicRange.min = -1 * xDynamicRange.max;

    setTensorQuantizationData(tensorIndex, SYN_QUANT_DYNAMIC_RANGE, &xDynamicRange, sizeof(synQuantDynamicRange));
}

TEST_F_GC(SynCastForNonTpcTest, mme_bf16_output_fp8_143_input_bf16_weights, {synDeviceGaudi2})
{
    castNonTpcApiTest(syn_type_fp8_143, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynCastForNonTpcTest, mme_bf16_output_bf16_input_fp8_143_weights, {synDeviceGaudi2})
{
    castNonTpcApiTest(syn_type_bf16, syn_type_fp8_143, syn_type_bf16);
}

TEST_F_GC(SynCastForNonTpcTest, mme_bf16_output_fp8_152_input_bf16_weights, {synDeviceGaudi2})
{
    castNonTpcApiTest(syn_type_fp8_152, syn_type_bf16, syn_type_bf16);
}

TEST_F_GC(SynCastForNonTpcTest, mme_bf16_output_bf16_input_fp8_152_weights, {synDeviceGaudi2})
{
    castNonTpcApiTest(syn_type_bf16, syn_type_fp8_152, syn_type_bf16);
}