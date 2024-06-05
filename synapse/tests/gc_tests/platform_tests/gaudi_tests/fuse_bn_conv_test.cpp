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

class SynFuseBNConvTest : public SynTrainingTestInfra
{
public:
    void fuseBNConvTest(synDataType xTensorDtype, synDataType wTensorDtype, synDataType yTensorDtype);
};

void SynFuseBNConvTest::fuseBNConvTest(synDataType xTensorDtype,
                                             synDataType wTensorDtype,
                                             synDataType yTensorDtype)
{
    // Tested Graph:      ____      ____
    //              x->->|    |    |    |
    //                   |Conv|->->| BN |->->y
    //              w->->|____|    |____|
    setGraphInferenceMode();
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

    const unsigned batch = 1;
    const unsigned nIFM  = 1;
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH);

    const unsigned ofmDataSize = nOFM * wOFM * hOFM * batch;

    // create_tensor's layout
    unsigned dims          = 4;
    unsigned ifmDimSizes[] = {nIFM, wIFM, hIFM, batch};
    unsigned wghDimSizes[] = {nOFM, nIFM, convParams.kW, convParams.kH};
    unsigned ofmDimSizes[] = {nOFM, wOFM, hOFM, batch};


    unsigned bn_sizes[] = {nOFM, 1, 1, 1, 1};

    int inputSize = nIFM * wIFM * hIFM * batch;
    float*         inputArray = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        inputArray[i] = (float)i * 0.5;
    }

    int weightsSize = nOFM * nIFM * convParams.kW * convParams.kH;
    float*         weightsArray = new float[weightsSize];
    for (int i = 0; i < weightsSize; i++)
    {
        weightsArray[i] = (float)i * 0.5;
    }

    float*         bnBetaArray = new float[nOFM];
    for (int i = 0; i < nOFM; i++)
    {
        bnBetaArray[i] = (float)i / 2;
    }

    float*         bnGammaArray = new float[nOFM];
    for (int i = 0; i < nOFM; i++)
    {
        bnGammaArray[i] = (float)(i + 1);
    }
    float ofmRefBuffer[ofmDataSize] = {91.5,  100.5, 109.5, 118.5,
                                       145.5, 154.5, 163.5, 172.5,
                                       199.5, 208.5, 217.5, 226.5,
                                       253.5, 262.5, 271.5, 280.5};

    unsigned convInTensor =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputArray, ifmDimSizes, dims, syn_type_float);

    unsigned convWeightTensor = createConstPersistTensor(INPUT_TENSOR,
                                                     MEM_INIT_FROM_INITIALIZER,
                                                     weightsArray,
                                                     wghDimSizes,
                                                     dims,
                                                     syn_type_float,
                                                     nullptr,
                                                     "constWeightTensor");
    unsigned convOutTensor =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_float);

    unsigned bnBetaTensor =
        createConstTensor(MEM_INIT_FROM_INITIALIZER, bnBetaArray, bn_sizes, 1, syn_type_float);

    unsigned bnGammaTensor =
        createConstTensor(MEM_INIT_FROM_INITIALIZER, bnGammaArray, bn_sizes, 1, syn_type_float);


    unsigned bnMeanTensor =
        createConstTensor(MEM_INIT_ALL_ZERO, nullptr, bn_sizes, 1, syn_type_float);

    unsigned bnVarTensor =
        createConstTensor(MEM_INIT_ALL_ONES, nullptr, bn_sizes, 1, syn_type_float);

    unsigned bnOutTensor =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_float);


    TensorIndices inputIndices  = {convInTensor, convWeightTensor};
    TensorIndices outputIndices = {convOutTensor};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&convParams,
                   sizeof(synConvolutionParams));

    TensorIndices bnInputIndices = {convOutTensor, bnBetaTensor, bnGammaTensor, bnMeanTensor, bnVarTensor};
    TensorIndices bnOutputIndices = {bnOutTensor};
    ns_BatchNormKernel::Params bnParams {};
    addNodeToGraph("batch_norm_inf_f32",
                   bnInputIndices,
                   bnOutputIndices,
                   &bnParams,
                   sizeof(bnParams));

    synDataType compareDtype = xTensorDtype;


    compileTopology();
    runTopology();

    float*      pOutputBuffer = (float*)m_hostBuffers[bnOutTensor];
    std::string errMsg;
    ASSERT_TRUE(
        compareFP8Results(ofmRefBuffer, pOutputBuffer, getNumberOfElements(ofmDimSizes), compareDtype, errMsg))
        << errMsg;
}

TEST_F_GC(SynFuseBNConvTest, fuseConvBN_f32, {synDeviceGaudi2})
{
    fuseBNConvTest(syn_type_float, syn_type_float, syn_type_float);
}

