#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include <algorithm>
#include <cstddef>
#include <data_types/bfloat16.h>
#include <iterator>
#include <numeric>

class SynMmeInferenceTest : public SynTrainingTestInfra {};

TEST_F_GC(SynMmeInferenceTest, mme_bf16_output_fp8_143_inputs)
{
    setGraphInferenceModeAndQuantizationEnabled();
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
    float ofmRefBuffer[ofmDataSize];

    // create_tensor's layout
    unsigned dims = 4;
    unsigned ifmDimSizes[] = { nIFM, wIFM, hIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmDimSizes[] = { nOFM, wOFM, hOFM, batch };

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, ifmDimSizes, dims, syn_type_float);
    unsigned wTensorIndex = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, wghDimSizes,
                                                     dims, syn_type_float, nullptr, "constWeightTensor");
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_float);

    unsigned xTensorIndexFp8  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ifmDimSizes, dims, syn_type_fp8_143);
    unsigned wTensorIndexFp8  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wghDimSizes, dims, syn_type_fp8_143);
    unsigned yTensorIndexBF16 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_bf16);

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];

    float xVal = ((float*)xData)[0];
    float xMaxVal = xVal;
    for (unsigned i = 0; i < getNumberOfElements(ifmDimSizes); i++)
    {
        xVal = abs(((float*)xData)[i]);
        xMaxVal = xMaxVal > xVal ? xMaxVal : xVal;
    }

    float wVal = ((float*)wData)[0];
    float wMaxVal = wVal;
    for (unsigned i = 0; i < getNumberOfElements(wghDimSizes); i++)
    {
        wVal = abs(((float*)wData)[i]);
        wMaxVal = wMaxVal > wVal ? wMaxVal : wVal;
    }

    synQuantDynamicRange xDynamicRange;
    xDynamicRange.max = xMaxVal + 1;
    xDynamicRange.min = -1 * xDynamicRange.max;

    synQuantDynamicRange wDynamicRange;
    wDynamicRange.max = wMaxVal + 1;
    wDynamicRange.min = -1 * wDynamicRange.max;

    setTensorQuantizationData(xTensorIndexFp8,
                              SYN_QUANT_DYNAMIC_RANGE,
                              &xDynamicRange,
                              sizeof(synQuantDynamicRange));

    setTensorQuantizationData(wTensorIndexFp8,
                              SYN_QUANT_DYNAMIC_RANGE,
                              &wDynamicRange,
                              sizeof(synQuantDynamicRange));

    // Adding cast node only for CPU conv calculations purposes
    addNodeToGraph("cast_f32_to_hf8",
                   {xTensorIndex},
                   {xTensorIndexFp8});
    addNodeToGraph("cast_f32_to_hf8",
                   {wTensorIndex},
                   {wTensorIndexFp8});
    addNodeToGraph("cast_bf16_to_f32",
                   {yTensorIndexBF16},
                   {yTensorIndex});

    TensorIndices inputIndices  = {xTensorIndexFp8, wTensorIndexFp8};
    TensorIndices outputIndices = {yTensorIndexBF16};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params, sizeof(synConvolutionParams));

    compileTopology();
    runTopology();

    // Calculate reference on CPU using HPU casted fp8 weight
    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    calculateFwdConvolution(xDesc, (char*)xData, wDesc, (char*)wData, yDesc, (char*)ofmRefBuffer, params, m_deviceType);

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    std::string errMsg;
    ASSERT_TRUE(compareFP8Results(ofmRefBuffer, pOutputBuffer, getNumberOfElements(ofmDimSizes), syn_type_fp8_143, errMsg)) << errMsg;
}

TEST_F_GC(SynMmeInferenceTest, mme_bf16_output_fp8_152_inputs)
{
    setGraphInferenceModeAndQuantizationEnabled();
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
    float ofmRefBuffer[ofmDataSize];

    // create_tensor's layout
    unsigned dims = 4;
    unsigned ifmDimSizes[] = { nIFM, wIFM, hIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmDimSizes[] = { nOFM, wOFM, hOFM, batch };

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, ifmDimSizes, dims, syn_type_float);
    unsigned wTensorIndex = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, wghDimSizes,
                                                     dims, syn_type_float, nullptr, "constWeightTensor");
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_float);

    unsigned xTensorIndexFp8  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ifmDimSizes, dims, syn_type_fp8_152);
    unsigned wTensorIndexFp8  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wghDimSizes, dims, syn_type_fp8_152);
    unsigned yTensorIndexBF16 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_bf16);

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];

    float xVal = ((float*)xData)[0];
    float xMaxVal = xVal;
    for (unsigned i = 0; i < getNumberOfElements(ifmDimSizes); i++)
    {
        xVal = abs(((float*)xData)[i]);
        xMaxVal = xMaxVal > xVal ? xMaxVal : xVal;
    }

    float wVal = ((float*)wData)[0];
    float wMaxVal = wVal;
    for (unsigned i = 0; i < getNumberOfElements(wghDimSizes); i++)
    {
        wVal = abs(((float*)wData)[i]);
        wMaxVal = wMaxVal > wVal ? wMaxVal : wVal;
    }

    synQuantDynamicRange xDynamicRange;
    xDynamicRange.max = xMaxVal + 1;
    xDynamicRange.min = -1 * xDynamicRange.max;

    synQuantDynamicRange wDynamicRange;
    wDynamicRange.max = wMaxVal + 1;
    wDynamicRange.min = -1 * wDynamicRange.max;

    setTensorQuantizationData(xTensorIndexFp8,
                              SYN_QUANT_DYNAMIC_RANGE,
                              &xDynamicRange,
                              sizeof(synQuantDynamicRange));

    setTensorQuantizationData(wTensorIndexFp8,
                              SYN_QUANT_DYNAMIC_RANGE,
                              &wDynamicRange,
                              sizeof(synQuantDynamicRange));

    // Adding cast node only for CPU conv calculations purposes
    addNodeToGraph("cast_f32_to_f8",
                   {xTensorIndex},
                   {xTensorIndexFp8});
    addNodeToGraph("cast_f32_to_f8",
                   {wTensorIndex},
                   {wTensorIndexFp8});
    addNodeToGraph("cast_bf16_to_f32",
                   {yTensorIndexBF16},
                   {yTensorIndex});

    TensorIndices inputIndices  = {xTensorIndexFp8, wTensorIndexFp8};
    TensorIndices outputIndices = {yTensorIndexBF16};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params, sizeof(synConvolutionParams));

    compileTopology();
    runTopology();

    // Calculate reference on CPU using HPU casted fp8 weight
    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    calculateFwdConvolution(xDesc, (char*)xData, wDesc, (char*)wData, yDesc, (char*)ofmRefBuffer, params, m_deviceType);

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    std::string errMsg;
    ASSERT_TRUE(compareFP8Results(ofmRefBuffer, pOutputBuffer, getNumberOfElements(ofmDimSizes), syn_type_fp8_152, errMsg)) << errMsg;
}

TEST_F_GC(SynMmeInferenceTest, mme_bf16_output_bf16_inputs)
{
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
    bfloat16 ofmRefBuffer[ofmDataSize];

    // create_tensor's layout
    unsigned dims = 4;
    unsigned ifmDimSizes[] = { nIFM, wIFM, hIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmDimSizes[] = { nOFM, wOFM, hOFM, batch };

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, ifmDimSizes, dims, syn_type_bf16);
    unsigned wTensorIndex = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, wghDimSizes,
                                                     dims, syn_type_bf16, nullptr, "constWeightTensor");
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_bf16);

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];
    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params, sizeof(synConvolutionParams));

    compileTopology();
    runTopology();

    // Calculate reference on CPU using HPU casted fp8 weight
    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    calculateFwdConvolution(xDesc, (char*)xData, wDesc, (char*)wData, yDesc, (char*)ofmRefBuffer, params, m_deviceType);

    bfloat16* pOutputBuffer = (bfloat16*)m_hostBuffers[yTensorIndex];
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes); i++)
    {
        ASSERT_LE(abs(pOutputBuffer[i] - ofmRefBuffer[i]), 0.00001)
            << "Mismatch at index " << i << " Result:" << pOutputBuffer[i] << " Ref: " << ofmRefBuffer[i];
    }
}

TEST_F_GC(SynMmeInferenceTest, mme_f32_output_f32_inputs)
{
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
    float ofmRefBuffer[ofmDataSize];

    // create_tensor's layout
    unsigned dims = 4;
    unsigned ifmDimSizes[] = { nIFM, wIFM, hIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmDimSizes[] = { nOFM, wOFM, hOFM, batch };

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, ifmDimSizes, dims, syn_type_float);
    unsigned wTensorIndex = createConstPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, wghDimSizes,
                                                     dims, syn_type_float, nullptr, "constWeightTensor");
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_float);

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];
    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params, sizeof(synConvolutionParams));

    compileTopology();
    runTopology();

    // Calculate reference on CPU using HPU casted fp8 weight
    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    calculateFwdConvolution(xDesc, (char*)xData, wDesc, (char*)wData, yDesc, (char*)ofmRefBuffer, params, m_deviceType);

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes); i++)
    {
        ASSERT_LE(abs(pOutputBuffer[i] - ofmRefBuffer[i]), 0.00001)
            << "Mismatch at index " << i << " Result:" << pOutputBuffer[i] << " Ref: " << ofmRefBuffer[i];
    }
}
