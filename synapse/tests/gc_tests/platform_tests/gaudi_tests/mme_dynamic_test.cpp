#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "data_type_utils.h"
#include "gc_dynamic_shapes_infra.h"
#include <math.h>
#include <random>

class SynTrainingMmeDynamicTest :
 public SynGaudiDynamicShapesTestsInfra,
 public testing::WithParamInterface<std::tuple<TestSizes    /* xSizes */,
                                               unsigned    /* yChannels */,
                                               unsigned    /* filter */,
                                               synDataType /* data type */,
                                               bool        /* dynamic */>>
{
    public:
    void runDeDwTest()
    {
        TestSizes xTensorSizes = std::get<0>(GetParam());
        unsigned yChannels = std::get<1>(GetParam());
        unsigned filter = std::get<2>(GetParam());
        synDataType dt = std::get<3>(GetParam());
        bool dynamic = std::get<4>(GetParam());

        synConvolutionParams params;
        params.kH = filter;
        params.kW = filter;
        TestSizes xTensorMinSizes = { xTensorSizes[0], xTensorSizes[1]/2, xTensorSizes[2]/2, xTensorSizes[3]/2, 1 };
        TestSizes xTensorActSizes = { xTensorSizes[0], xTensorSizes[1]*2/3, xTensorSizes[2]*2/3, xTensorSizes[3]*2/3, 1 };

        TestSizes wTensorSizes = {yChannels, xTensorSizes[0], params.kW, params.kH, 1};

        TestSizes yTensorSizes = {yChannels,
                                  convOutputDimSize(xTensorSizes[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
                                  convOutputDimSize(xTensorSizes[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
                                  xTensorSizes[3], 1};

        TestSizes yTensorMinSizes = {yChannels,
                                  convOutputDimSize(xTensorSizes[1]/2, params.kW, params.dW, params.padL + params.padR, params.dilW),
                                  convOutputDimSize(xTensorSizes[2]/2, params.kH, params.dH, params.padT + params.padB, params.dilH),
                                  xTensorSizes[3]/2, 1};

        TestSizes yTensorActSizes = {yChannels,
                                  convOutputDimSize(xTensorSizes[1]*2/3, params.kW, params.dW, params.padL + params.padR, params.dilW),
                                  convOutputDimSize(xTensorSizes[2]*2/3, params.kH, params.dH, params.padT + params.padB, params.dilH),
                                  xTensorSizes[3]*2/3, 1};

        std::vector<float> xInitData((size_t)xTensorSizes[0]*xTensorSizes[1]*xTensorSizes[1]*xTensorSizes[3]);
        std::vector<float> yInitData((size_t)yTensorSizes[0]*yTensorSizes[1]*yTensorSizes[1]*yTensorSizes[3]);

        std::default_random_engine e;
        std::uniform_real_distribution<float>     dist(-100, 100);
        for (int i = 0; i < xInitData.size() || i < yInitData.size(); ++i)
        {
            if (i < xInitData.size()) xInitData[i] = dist(e);
            if (i < yInitData.size()) yInitData[i] = dist(e);
        }

        if (!dynamic)
        {
            xTensorSizes = (xTensorMinSizes = xTensorActSizes);
            yTensorSizes = (yTensorMinSizes = yTensorActSizes);
        }


        unsigned yTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, yInitData.data(),
                                               yTensorSizes.data(), 4, dt,
                                               nullptr, nullptr, 0, 0, nullptr, yTensorMinSizes.data());

        unsigned xTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, xInitData.data(),
                                               xTensorSizes.data(), 4, dt,
                                               nullptr, nullptr, 0, 0, nullptr, xTensorMinSizes.data());

        unsigned wTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               wTensorSizes.data(), 4, syn_type_float);

        addNodeToGraph("dedw", {yTensor, xTensor}, {wTensor}, &params, sizeof(synConvolutionParams));
        compileTopology("gaudi_dedw");

        if (dynamic)
        {
            setActualSizes(yTensor, yTensorActSizes.data());
            setActualSizes(xTensor, xTensorActSizes.data());
            // update tensor desc - so that reference will work on actual sizes.
            castNcopy(m_tensorDescs[yTensor].m_sizes, yTensorActSizes, sizeof(unsigned) * m_tensorDescs[yTensor].m_dims);
            castNcopy(m_tensorDescs[xTensor].m_sizes, xTensorActSizes, sizeof(unsigned) * m_tensorDescs[xTensor].m_dims);
        }
        runTopology();

        const auto yDesc           = static_cast<synTensorDescriptor>(m_tensorDescs[yTensor]);
        const auto xDesc           = static_cast<synTensorDescriptor>(m_tensorDescs[xTensor]);
        char*      xData           = (char*)m_hostBuffers[xTensor];
        char*      yData           = (char*)m_hostBuffers[yTensor];
        const auto wDesc           = static_cast<synTensorDescriptor>(m_tensorDescs[wTensor]);
        unsigned wSizeInElements = 1;
        for (unsigned i = 0; i < wDesc.m_dims; i++)
        {
            wSizeInElements *= wTensorSizes[i];
        }
        float *wData = new float[wSizeInElements];

        calculateDEDW(yDesc, (char*)yData, xDesc, xData, wDesc, (char*)wData, params, m_deviceType);

        float *result = (float *) m_hostBuffers[wTensor];
        validateResult(wData, result, wSizeInElements);
        delete[] wData;
    }
};

class SynDedwDynamicTest : public SynTrainingMmeDynamicTest
{
};

INSTANTIATE_TEST_SUITE_P(dedw_dynamic_test,
                         SynDedwDynamicTest,
                         ::testing::Combine(::testing::Values<TestSizes, TestSizes>({12,6,6, 12,1}, {12*5, 6*5, 6*5, 12*5,1}), /* xTensorSizes*/
                                            ::testing::Values(8),                             /* yChannels */
                                            ::testing::Values(1),                             /* Filter */
                                            ::testing::Values(syn_type_bf16, syn_type_float), /* data type */
                                            ::testing::Values(true, false)                    /* dynamic */),
                         [](const testing::TestParamInfo<SynTrainingMmeDynamicTest::ParamType>& info)
                         {
                            std::string dtype = std::get<3>(info.param) == syn_type_bf16 ? "bf16" : "float";
                            std::string dynamic = std::get<4>(info.param) == true ? "dynamic" : "static";
                            std::string bigSmall = std::get<0>(info.param)[1] < 10 ? "small" : "big";
                            return dtype + "_" + bigSmall + "_" + dynamic;
                         }
                         );


TEST_P_GC(SynDedwDynamicTest, dedw_dynamic_tests)
{
    runDeDwTest();
}
