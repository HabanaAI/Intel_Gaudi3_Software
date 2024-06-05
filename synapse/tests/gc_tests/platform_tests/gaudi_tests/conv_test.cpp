#include "infra/cpu_calculator.h"
#include "utils.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"

class SynTrainingConvTest : public SynTrainingTestInfra
{
public:
    SynTrainingConvTest() { setTestPackage(TEST_PACKAGE_CONVOLUTION); }
    void transformToFCDBuffer(unsigned spacial,
                              const unsigned channels,
                              float* inputBuffer,
                              float* FCDBuffer);

    void reluConvRelu(const unsigned consecutiveConvCnt = 1, const unsigned parallelSubGraphs = 1,
                      const unsigned kernelSize = 2, const unsigned kernelCnt = 1,
                      const unsigned inC = 20, const unsigned inW = 150,
                      const unsigned inH = 150, const unsigned inB = 10);
};

/*
 * Transform input buffer to be ordered by FCD.
 * Example:
 *  1, 2, 3
 *  3, 3, 2,
 *  2, 5, 8,
 *
 *  2, 3, 1,
 *  4, 1, 6,
 *  5, 9, 8,
 *
 *  3, 1, 2,
 *  5, 3, 1,
 *  7, 1, 2
 *
 *  should be transformed into
 *  1, 2, 3,
 *  2, 3, 1,
 *  3, 1, 2,
 *
 *  3, 4, 5,
 *  3, 1, 3,
 *  2, 6, 1,
 *
 *  2, 5, 7,
 *  5, 9, 1,
 *  8, 8, 2
 */
void SynTrainingConvTest::transformToFCDBuffer(unsigned       spacial,
                                               const unsigned channels,
                                               float*         inputBuffer,
                                               float*         FCDBuffer)
{
    for (int i = 0; i < channels; i++)
    {
        int startSpacialIndex = i * spacial;

        for (int j = 0; j < spacial; j++)
        {
            int index = (j * channels) + i;
            FCDBuffer[index] = inputBuffer[startSpacialIndex + j];
        }
    }
}

TEST_F_GC(SynTrainingConvTest, basic_L2)
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
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    float ifmBuffer[] =
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

    float wghBuffer[]
            {
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

    /* Input and weights are expected sorted by the FCD */
    const unsigned ifmDataSize = wIFM * hIFM * nIFM * batch;
    float* ifmBufferFCD = new float[ifmDataSize];
    unsigned spacial = wIFM * hIFM;
    transformToFCDBuffer(spacial, nIFM, ifmBuffer, ifmBufferFCD);

    const unsigned wghDataSize = params.kW * params.kH * nIFM * batch;
    float* wghBufferFCD = new float[wghDataSize];
    spacial = params.kW * params.kH;
    transformToFCDBuffer(spacial, nIFM, wghBuffer, wghBufferFCD);

    const unsigned ofmDataSize = nOFM * wOFM * hOFM * batch;
    float* ofmRefBuffer = new float[ofmDataSize];

    // create_tensor's layout
    unsigned dims = 4;
    unsigned ifmDimSizes[] = { nIFM, wIFM, hIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmDimSizes[] = { nOFM, wOFM, hOFM, batch };

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmBufferFCD, ifmDimSizes, dims, syn_type_single);
    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, wghBufferFCD, wghDimSizes, dims, syn_type_single);
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_single);

    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];

    calculateFwdConvolution(xDesc, (char*)xData, wDesc, (char*)wData, yDesc, (char*)ofmRefBuffer, params, m_deviceType);

    TensorIndices inputIndices = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params, sizeof(synConvolutionParams));

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes); i++)
    {
        ASSERT_EQ(*pOutputBuffer, ofmRefBuffer[i] ) << "Mismatch at index " << i
                                                   << " Result:"           << *pOutputBuffer
                                                   << " Ref: "             << ofmRefBuffer[i];
        pOutputBuffer++;
    }

    delete[] ifmBufferFCD;
    delete[] wghBufferFCD;
    delete[] ofmRefBuffer;
}

TEST_F_GC(SynTrainingConvTest, DISABLED_basic_bfloat16)
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
    const unsigned nIFM  = 1;
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    const unsigned ofmDataSize = nOFM * wOFM * hOFM * batch;

    float ifmBuffer[] =
        {
          0.5, 1, 1, 1, 2, 3,
            1, 1, 1, 2, 3, 1,
            1.2, 1.8, 1.06, 3.05, 1, 2,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2.5, 2,
            3, 3, 3, 3, 3, 3,
        };


    float wghBuffer[]
        {
            1, 1.2, 1.6,
            1, 1.5, 1.3,
            1.7, 1.03, 1.9,
        };

    float ofmRefBuffer[ofmDataSize];

    synActivationParams activationParams;
    ConvQuantizationParams qParams;
    ASSERT_EQ(float(bfloat16(277.f)), 276);

    DoConvolution2D<float,
                    float,
                    float,
                    float,
                    float>(ifmBuffer, wghBuffer, nullptr, nullptr, ofmRefBuffer,
                          wIFM, hIFM, nIFM,
                          wOFM, hOFM, nOFM,
                          wIFM, hIFM,
                          params.padL, params.padT,
                          params.kW, params.kH,
                          params.dW, params.dH,
                          params.dilW, params.dilH,
                          batch,
                          &activationParams,
                          &qParams);

    unsigned ifmDimSizes[] = { nIFM, wIFM, hIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmDimSizes[] = { nOFM, wOFM, hOFM, batch };
    unsigned ifmTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmBuffer, ifmDimSizes, syn_type_bf16);
    unsigned wghTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, wghBuffer, wghDimSizes, syn_type_bf16);
    unsigned ofmTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, syn_type_bf16);

    TensorIndices inputIndices = {ifmTensor, wghTensor};
    TensorIndices outputIndices = {ofmTensor};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params, sizeof(synActivationParams));

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[ofmTensor];

    for( unsigned i = 0; i< ofmDataSize; i++ )
    {
        // Verfiy DoConvolution2D vs bf16 MME
        ASSERT_LT(abs(float(pOutputBuffer[i]) - ofmRefBuffer[i]), 0.1)
            << "Mismatch at index " << i << " Result:(" << pOutputBuffer[i] << ", " << float(pOutputBuffer[i])
            << ") Ref: (" << bfloat16(ofmRefBuffer[i]) << ", " << ofmRefBuffer[i] << ")";
    }
}

// create 1 or 2 sub-graphs of relu->conv->relu.
// consecutiveConvCnt - specify how many conv+relu in each sub-graph. supported only for 1x1 kernels
// parallelSubGraphs - this will specify if we have 1 or 2 sub-graphs.
// kernelSize - in this function we assume kernel size = kernel width = kernel height
// kernelCnt - the number of kernels = the number of channels in the output
// inC - number of input channels
// inW - input width
// inH - input height
// inB - input batch size
void SynTrainingConvTest::reluConvRelu(const unsigned consecutiveConvCnt /*= 1*/,
                                       const unsigned parallelSubGraphs /*= 1*/,
                                       const unsigned kernelSize /*= 2*/,
                                       const unsigned kernelCnt /*= 1*/,
                                       const unsigned inC /*= 20*/,
                                       const unsigned inW /*= 150*/,
                                       const unsigned inH /*= 150*/,
                                       const unsigned inB /*= 10*/)
{
    ASSERT_TRUE(1 <= parallelSubGraphs && parallelSubGraphs <= 2) << "support only 1 or 2 sub-graphs";
    ASSERT_TRUE(consecutiveConvCnt == 1 || kernelSize == 1) << "consecutive convolutions is only supported";

    // init SRAM capacity to be small enough for us to have "interesting" graphs
    const uint64_t maxParamSize = 32;
    char originalSramCapacity[maxParamSize];
    synConfigurationGet("SRAM_SLICER_MAX_CAPACITY_BYTES", originalSramCapacity, maxParamSize);
    std::stringstream slicerCapVal;
    slicerCapVal << 8*1024*1024;
    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", slicerCapVal.str().c_str());

    //------------       init parameters dimensions:      ------------------------------------------------------------/
    synConvolutionParams params;
    params.kH = kernelSize, params.kW = kernelSize; // kernel size

    enum Dims             {  C,   W,   H,    B };
    unsigned inSizes[] =  { inC, inW, inH, inB };
    unsigned outSizes[] = { kernelCnt, inSizes[W] - kernelSize + 1, inSizes[H] - kernelSize + 1, inSizes[B] };
    //                              K          C          S           R
    unsigned weightsSizes[] = { kernelCnt, inSizes[C], params.kW , params.kH };

    //-----------        init buffers:          ----------------------------------------------------------------------/
    // weight input -same for all
    const unsigned wghElementCount = multiplyElements(std::begin(weightsSizes), std::end(weightsSizes));
    std::vector<float> wghBuffer(wghElementCount);
    fillBufferWithRunningNumbers(wghBuffer.data(), wghElementCount, 10, 1, 1);
    // A inputs - different per sub-grpah
    const unsigned ifmElementCount = multiplyElements(std::begin(inSizes), std::end(inSizes));

    std::vector<std::vector<float>> convInputs(parallelSubGraphs);

    for (int i = 0; i < parallelSubGraphs; ++i)
    {
        convInputs[i].resize(ifmElementCount);
        // we use "i" as offset so each of the sub-graphs will have different convolution inputs
        fillBufferWithRunningNumbers(convInputs[i].data(), ifmElementCount, 1000, 0.01, i, 3);
    }

    //------------       create graph and compile:         -----------------------------------------------------------/
    unsigned outputIndexes[parallelSubGraphs];
    unsigned convInputAIndexes[parallelSubGraphs];
    unsigned convInputBIndexes[parallelSubGraphs];
    for (int i = 0; i < parallelSubGraphs; ++i) // create 1 or 2 sub-graphs with relu+(conv+relu)*(x times)
    {
        // relu
        unsigned reluBeforeInIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, convInputs[i].data(),
                                                         inSizes);
        unsigned reluBeforeOutIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inSizes);
        addNodeToGraph("relu_fwd_f32", {reluBeforeInIndex}, {reluBeforeOutIndex});
        // conv+relu 1 or more times:
        unsigned connectorTensorIndex = reluBeforeOutIndex;
        for (int j = 1; j <= consecutiveConvCnt; ++j)
        {
            // conv
            unsigned xTensorIndex = connectOutputTensorToInputTensor(connectorTensorIndex);
            unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, wghBuffer.data(),
                                                        weightsSizes);
            unsigned yTensorIndex = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes);
            addNodeToGraph(NodeFactory::convolutionNodeTypeName, {xTensorIndex, wTensorIndex}, {yTensorIndex},
                           (void *) &params, sizeof(synConvolutionParams));
            if (j == 1)
            {
                convInputAIndexes[i] = connectorTensorIndex;
                convInputBIndexes[i] = wTensorIndex;
            }

            // relu
            unsigned reluAftereInIndex = connectOutputTensorToInputTensor(yTensorIndex);
            unsigned reluAfterOutIndex = -1;
            if (j < consecutiveConvCnt)
            {
                reluAfterOutIndex = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes);
            }
            else
            {
                reluAfterOutIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes);
            }
            addNodeToGraph("relu_fwd_f32", {reluAftereInIndex}, {reluAfterOutIndex});
            connectorTensorIndex = reluAfterOutIndex;
        }
        outputIndexes[i] = connectorTensorIndex;
    }
    compileAndRun();

    //------------       validate results:         -------------------------------------------------------------------/
    for (int i = 0; i < parallelSubGraphs; ++i)
    {
        synTensorDescriptor opADesc = m_tensorDescs[convInputAIndexes[i]];
        synTensorDescriptor opBDesc = m_tensorDescs[convInputBIndexes[i]];
        synTensorDescriptor opOutDesc = m_tensorDescs[outputIndexes[i]];
        void* opAData = m_hostBuffers[convInputAIndexes[i]];
        void* opBData = m_hostBuffers[convInputBIndexes[i]];
        void* opOutData = m_hostBuffers[outputIndexes[i]];

        //validate relu
        std::vector<float> ifmBufferRelued(ifmElementCount);
        memcpy(ifmBufferRelued.data(), convInputs[i].data(), ifmElementCount * sizeof(ifmBufferRelued.data()[0]));
        calculateRelu(ifmBufferRelued.data(), ifmElementCount);
        ASSERT_EQ(0, memcmp(&ifmBufferRelued.data()[0], opAData, ifmElementCount * sizeof(ifmBufferRelued.data()[0])));

        //validate convolution
        CoordArray wrongIdx = {0};
        bool       ret      = checkFwdConvolution(opADesc,
                                       static_cast<char*>(opAData),
                                       opBDesc,
                                       static_cast<char*>(opBData),
                                       opOutDesc,
                                       static_cast<char*>(opOutData),
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

    synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", originalSramCapacity);
}

TEST_F_GC(SynTrainingConvTest, relu_conv1x1_relu_L2)
{
    reluConvRelu(1, 1, 1, 1, 20, 150, 150, 10);
}

TEST_F_GC(SynTrainingConvTest, relu_conv1x1_relu_multiple_blobs_per_engine)
{
    reluConvRelu(1, 1, 1, 1, 20, 150, 150, 10);
}

// relu->conv2x2->relu
TEST_F_GC(SynTrainingConvTest, relu_conv2x2_relu_L2)
{
    reluConvRelu(1, 1, 2, 10);
}

// relu->conv3x3->relu
TEST_F_GC(SynTrainingConvTest, relu_conv3x3_relu)
{
    reluConvRelu(1, 1, 3, 1, 20, 100, 10, 1);
}

// relu->conv4x4->relu
TEST_F_GC(SynTrainingConvTest, relu_conv4x4_relu_L2)
{
    reluConvRelu(1, 1, 4, 1, 10, 15, 15, 2000);
}

// relu->conv5x5->relu
TEST_F_GC(SynTrainingConvTest, relu_conv5x5_relu_L2)
{
    reluConvRelu(1, 1, 5, 1, 1000, 15, 15, 10);
}

// relu->conv1x1->relu
// (parallel)
// relu->conv1x1->relu
TEST_F_GC(SynTrainingConvTest, relu_conv1x1_relu_twice_L2)
{
    reluConvRelu(1, 2, 1, 1, 1000, 2500, 1, 1);
}

// relu->conv2x2->relu
// (parallel)
// relu->conv2x2->relu
TEST_F_GC(SynTrainingConvTest, relu_conv2x2_relu_twice_L2)
{
    reluConvRelu(1, 2, 2, 1, 500, 15, 15, 10);
}

// relu->conv3x3->relu
// (parallel)
// relu->conv3x3->relu
TEST_F_GC(SynTrainingConvTest, relu_conv3x3_relu_twice_L2)
{
    reluConvRelu(1, 2, 3, 1);
}

// relu->conv4x4->relu
// (parallel)
// relu->conv4x4->relu
TEST_F_GC(SynTrainingConvTest, relu_conv4x4_relu_twice_L2)
{
    reluConvRelu(1, 2, 4, 1);
}

// relu->conv5x5->relu
// (parallel)
// relu->conv5x5->relu
TEST_F_GC(SynTrainingConvTest, relu_conv5x5_relu_twice_L2)
{
    reluConvRelu(1, 2, 5, 1);
}

// relu->conv1x1->relu->conv1x1->relu
// (parallel)
// relu->conv1x1->relu->conv1x1->relu
TEST_F_GC(SynTrainingConvTest, relu_conv1x1_relu_conv1x1_relu_twice_L2)
{
    reluConvRelu(2, 2, 1, 1, 1, 100, 100, 10);
}
