#include "infra/cpu_calculator.h"
#include "utils.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"

class SynGaudiConvWithAllowedPermutationTest : public SynGaudiTestInfra
{
public:
    void transformToFCDBuffer(unsigned                  spacial,
                              unsigned                  inChannels,
                              unsigned                  outChannels,
                              const std::vector<float>& inputBuffer,
                              std::vector<float>&       FCDBuffer);
};

void SynGaudiConvWithAllowedPermutationTest::transformToFCDBuffer(unsigned                  spacial,
                                                                  unsigned                  inChannels,
                                                                  unsigned                  outChannels,
                                                                  const std::vector<float>& inputBuffer,
                                                                  std::vector<float>&       FCDBuffer)
{
    for (int k = 0; k < outChannels; k++)
    {
        for (int i = 0; i < inChannels; i++)
        {
            int startSpacialIndex = (k * inChannels + i) * spacial;
            for (int j = 0; j < spacial; j++)
            {
                int index        = (j * inChannels + i) * outChannels + k;
                FCDBuffer[index] = inputBuffer[startSpacialIndex + j];
            }
        }
    }
}

TEST_F_GC(SynGaudiConvWithAllowedPermutationTest, basic_L2)
{
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
    const unsigned nIFM  = 3;
    const unsigned nOFM  = 2;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;
    const char*    outputName = "out";

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    std::vector<float> ifmBuffer =
        {
            1, 1, 2, 2, 3, 3,
            1, 1, 2, 2, 3, 3,
            4, 3, 1, 3, 2, 1,
            2, 6, 2, 0, 0, 0,
            1, 5, 2, 0, 0, 0,
            4, 6, 5, 0, 0, 0,

            2, 2, 3, 3, 1, 1,
            1, 1, 2, 2, 3, 3,
            5, 1, 5, 2, 2, 5,
            2, 4, 3, 0, 0, 0,
            3, 1, 2, 0, 0, 0,
            3, 6, 4, 0, 0, 0,

            3, 3, 1, 1, 2, 2,
            1, 1, 2, 2, 3, 3,
            1, 1, 4, 1, 3, 1,
            1, 2, 5, 0, 0, 0,
            3, 5, 1, 0, 0, 0,
            6, 4, 8, 0, 0, 0
        };

    std::vector<float> wghBuffer =
        {
            1, 2, 3,
            3, 3, 2,
            2, 5, 8,

            2, 3, 1,
            4, 1, 6,
            5, 9, 8,

            3, 1, 2,
            5, 3, 1,
            7, 1, 2,

            1, 2, 3,
            3, 3, 2,
            2, 5, 8,

            2, 3, 1,
            4, 1, 6,
            5, 9, 8,

            3, 1, 2,
            5, 3, 1,
            7, 1, 2
        };

    // calculate the output using the device
    unsigned dims          = 4;
    unsigned ifmDimSizes[] = {wIFM, hIFM, nIFM, batch};
    unsigned wghDimSizes[] = {params.kW, params.kH, nIFM, nOFM};
    unsigned ofmDimSizes[] = {wOFM, hOFM, nOFM, batch};

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                ifmBuffer.data(),
                                                ifmDimSizes,
                                                dims,
                                                syn_type_single);
    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                wghBuffer.data(),
                                                wghDimSizes,
                                                dims,
                                                syn_type_single);
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                ofmDimSizes,
                                                dims,
                                                syn_type_single,
                                                nullptr,
                                                outputName);

    synTensorSetAllowPermutation(m_tensors[yTensorIndex], 1);

    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    const char* conv2D_in_layouts[]  = {"WHCN", "SRCK"};
    const char* conv2D_out_layouts[] = {"WHCN"};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params,
                   sizeof(synConvolutionParams),
                   nullptr,
                   0,
                   nullptr,
                   conv2D_in_layouts,
                   conv2D_out_layouts);

    compileAndRun();

    // find tensor metadata
    GraphData& graphData    = getGraph(0);
    uint32_t   numOfTensors = 0;

    ASSERT_EQ(synSuccess, synTensorRetrieveLaunchAmount(graphData.recipeHandle, &numOfTensors));
    ASSERT_EQ(3, numOfTensors);

    uint64_t ids[numOfTensors];

    ASSERT_EQ(synSuccess, synTensorRetrieveLaunchIds(graphData.recipeHandle, ids, numOfTensors));
    synRetrievedLaunchTensorInfo tensorInfos[numOfTensors];
    for (unsigned i = 0; i < numOfTensors; i++)
    {
        tensorInfos[i].tensorId = ids[i];
    }
    ASSERT_EQ(synSuccess, synTensorRetrieveLaunchInfoById(graphData.recipeHandle, numOfTensors, tensorInfos));
    unsigned outIdx = 0;
    for (; outIdx < numOfTensors; outIdx++)
    {
        if (strcmp(tensorInfos[outIdx].tensorName, outputName) == 0) break;
    }
    const synRetrievedLaunchTensorInfo& outputTensorInfo = tensorInfos[outIdx];

    // verify the permutation is as expected from the recipe
    ASSERT_EQ(outputTensorInfo.tensorDims, dims);
    uint8_t expectedPermutation[] = {2, 0, 1, 3};
    for (unsigned i = 0; i < outputTensorInfo.tensorDims; i++)
    {
        ASSERT_EQ(outputTensorInfo.tensorPermutation[i], expectedPermutation[i]);
    }

    // calculate the output using the CPU

    const unsigned     ifmDataSize = wIFM * hIFM * nIFM * batch;
    std::vector<float> ifmFCDBuffer(ifmDataSize);
    unsigned           ifmFCDspacial = wIFM * hIFM;
    transformToFCDBuffer(ifmFCDspacial, nIFM, 1, ifmBuffer, ifmFCDBuffer);

    const unsigned     wghDataSize = params.kW * params.kH * nIFM * nOFM;
    std::vector<float> wghFCDBuffer(wghDataSize);
    unsigned           wghFCDSpacial = params.kW * params.kH;
    transformToFCDBuffer(wghFCDSpacial, nIFM, nOFM, wghBuffer, wghFCDBuffer);

    const unsigned     ofmDataSize = nOFM * wOFM * hOFM * batch;
    std::vector<float> expectedFCDOutputBuffer(ofmDataSize);

    unsigned ifmFCDDimSizes[] = {nIFM, wIFM, hIFM, batch};
    unsigned wghFCDDimSizes[] = {nOFM, nIFM, params.kW, params.kH};
    unsigned ofmFCDDimSizes[] = {nOFM, wOFM, hOFM, batch};

    unsigned xFCDTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                   MEM_INIT_FROM_INITIALIZER,
                                                   ifmFCDBuffer.data(),
                                                   ifmFCDDimSizes,
                                                   dims,
                                                   syn_type_single);
    unsigned wFCDTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                   MEM_INIT_FROM_INITIALIZER,
                                                   wghFCDBuffer.data(),
                                                   wghFCDDimSizes,
                                                   dims,
                                                   syn_type_single);
    unsigned yFCDTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmFCDDimSizes, dims, syn_type_single);

    synTensorDescriptor xFCDDesc = m_tensorDescs[xFCDTensorIndex];
    synTensorDescriptor wFCDDesc = m_tensorDescs[wFCDTensorIndex];
    synTensorDescriptor yFCDDesc = m_tensorDescs[yFCDTensorIndex];

    auto xFCDData = m_hostBuffers[xFCDTensorIndex];
    auto wFCDData = m_hostBuffers[wFCDTensorIndex];

    calculateFwdConvolution(xFCDDesc,
                            (char*)xFCDData,
                            wFCDDesc,
                            (char*)wFCDData,
                            yFCDDesc,
                            (char*)expectedFCDOutputBuffer.data(),
                            params,
                            m_deviceType);

    // compare the two outputs (CPU and device)
    float* pOutputBuffer     = static_cast<float*>(m_hostBuffers[yTensorIndex]);
    int    outputSpacialSize = wOFM * hOFM;
    for (int i = 0; i < nOFM; i++)
    {
        for (int j = 0; j < outputSpacialSize; j++)
        {
            int indexFCD = (j * nOFM) + i;
            ASSERT_EQ(pOutputBuffer[indexFCD], expectedFCDOutputBuffer[indexFCD])
                << "Mismatch at Spatial index " << j << " channel index: " << i << " Result:" << pOutputBuffer[indexFCD]
                << " Ref: " << expectedFCDOutputBuffer[indexFCD];
        }
    }
}