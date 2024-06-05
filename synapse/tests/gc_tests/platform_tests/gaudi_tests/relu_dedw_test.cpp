#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include <math.h>

class SynTrainingReluDeDwTest : public SynTrainingTestInfra
{
    public:
    void runReluDeDwTest(TestSizes xTensorSizes, unsigned yChannels, unsigned H, unsigned W)
    {
        synConvolutionParams params;
        params.kH = H;
        params.kW = W;

        TestSizes wTensorSizes = {yChannels, xTensorSizes[0], params.kW, params.kH, 1};
        TestSizes yTensorSizes = {yChannels,
                                  convOutputDimSize(xTensorSizes[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
                                  convOutputDimSize(xTensorSizes[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
                                  xTensorSizes[3], 1};

        unsigned yTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               yTensorSizes.data(), 4, syn_type_bf16);
        unsigned yTensorRelu = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, yTensorSizes.data(),
                                            4, syn_type_bf16);
        unsigned yInputToDeDw = connectOutputTensorToInputTensor(yTensorRelu);
        unsigned xTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               xTensorSizes.data(), 4, syn_type_bf16);
        unsigned wTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               wTensorSizes.data(), 4, syn_type_float);

        addNodeToGraph("relu_fwd_bf16", {yTensor}, {yTensorRelu});
        addNodeToGraph("dedw", {yInputToDeDw, xTensor}, {wTensor}, &params, sizeof(synConvolutionParams));
        compileTopology("gaudi_relu_dedw");
        runTopology();

        synTensorDescriptor yDesc = m_tensorDescs[yTensor];
        synTensorDescriptor xDesc = m_tensorDescs[xTensor];
        char *xData = (char *) m_hostBuffers[xTensor];
        synTensorDescriptor wDesc = m_tensorDescs[wTensor];
        unsigned wSizeInElements = 1;
        unsigned ySizeInElements = 1;
        for (unsigned i = 0; i < wDesc.m_dims; i++)
        {
            wSizeInElements *= wTensorSizes[i];
        }
        float *wData = new float[wSizeInElements];

        for (unsigned i = 0; i < yDesc.m_dims; i++)
        {
            ySizeInElements *= yTensorSizes[i];
        }
        bfloat16 *yData = new bfloat16[ySizeInElements];
        memcpy(yData, m_hostBuffers[yTensor], ySizeInElements * sizeof(uint16_t));
        calculateRelu(yData, ySizeInElements);
        calculateDEDW(yDesc, (char*)yData, xDesc, xData, wDesc, (char*)wData, params, m_deviceType);
        float *result = (float *) m_hostBuffers[wTensor];
        validateResult(wData, result, wSizeInElements);
        delete[] wData;
        delete[] yData;
    }
};

TEST_F_GC(SynTrainingReluDeDwTest, bf16_relu_dedw_1_1_test_ASIC)
{
    TestSizes xTensorSizes = {64, 56, 56, 64, 1};
    runReluDeDwTest(xTensorSizes, 256, 1, 1);
}

TEST_F_GC(SynTrainingReluDeDwTest, bf16_relu_dedw_3_3_test_ASIC)
{
    TestSizes xTensorSizes = {64, 56, 56, 64, 1};
    runReluDeDwTest(xTensorSizes, 64, 3, 3);
}
