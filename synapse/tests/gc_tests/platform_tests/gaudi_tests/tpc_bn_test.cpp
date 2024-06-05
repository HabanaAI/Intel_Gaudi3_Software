#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "log_manager.h"
#include <math.h>
#include "tpc_batch_norm_test.inl"
#include <fstream>

#define MAT_SIZE 25

float relu(float k)
{
    return (k < 0 ) ? 0 : k;
}
bfloat16 relu(bfloat16 k)
{
    return (k < 0.0f) ? bfloat16(0.0f) : k;
}

class SynTrainingBNTest : public SynTrainingTestInfra
{
public:
    template<typename DATA_TYPE, bool isTraining>
    void batch_norm_fwd_test();

    template<typename DATA_TYPE>
    void moments_simple_test();

    template<typename DATA_TYPE>
    void moments_random_input_test();
};

template<typename DATA_TYPE, bool isTraining>
void SynTrainingBNTest::batch_norm_fwd_test()
{
    ns_BatchNormKernel::ParamsV2 bnParams;
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon     = 1e-05;
    bnParams.isTraining  = isTraining;

    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };
    float runningMean        = 0;
    float runningVar         = isTraining ? 1 : 2;  // must always be 1 for training

    float beta = 0;
    float gamma = 1;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    synDataType synDType = std::is_same<DATA_TYPE, float>::value ? syn_type_float : syn_type_bf16;
    std::string guid = std::is_same<DATA_TYPE, float>::value ? "batch_norm_fwd_f32" : "batch_norm_fwd_bf16";


    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,synDType, nullptr, "featureMapIn");
    unsigned betaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float, nullptr, "betaIn");
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes,1 ,syn_type_float, nullptr, "gammaIn");
    unsigned runningMeanIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningMean, oneDimSizes,1, syn_type_float, nullptr, "runningMeanIn");
    unsigned runningVarIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningVar, oneDimSizes,1, syn_type_float, nullptr, "runningVarIn");

    unsigned featureMapOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, synDType, nullptr, "featureMapOut");
    unsigned savedMeanOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1,  syn_type_float, nullptr, "savedMeanOut");
    unsigned iStdOut        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float, nullptr, "iStdOut");

    addNodeToGraph(guid.c_str(),
                   {featureMapIn, betaIn, gammaIn, runningMeanIn, runningVarIn},
                   {featureMapOut, savedMeanOut, iStdOut},
                   &bnParams,
                   sizeof(bnParams));

    compileTopology();
    runTopology();

    DATA_TYPE* pFmOutput = (DATA_TYPE*)m_hostBuffers[featureMapOut];
    DATA_TYPE* pFmInput = (DATA_TYPE*)m_hostBuffers[featureMapIn];

    float mean = 0;
    float iStd = 1;
    DATA_TYPE outRef[MAT_SIZE] = {};
    calcBatchNormForwardRef<DATA_TYPE>(&mean, &iStd, &runningMean, &runningVar, outRef, 0.1, pFmInput, &beta, &gamma, fMsizes);
    validateResult(outRef, pFmOutput, MAT_SIZE);
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_test_training)
{
    batch_norm_fwd_test<bfloat16, true>();
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_test_no_training)
{
    batch_norm_fwd_test<bfloat16, false>();
}

TEST_F_GC(SynTrainingBNTest, f32_batch_norm_test_training)
{
    batch_norm_fwd_test<float, true>();
}

TEST_F_GC(SynTrainingBNTest, f32_batch_norm_test_no_training)
{
    batch_norm_fwd_test<float, false>();
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_relu_fusion_test_L2)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };

    float runningMean = 0;
    float runningVar = 1;

    float beta = 0;
    float gamma = 1;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,syn_type_bf16);
    unsigned betaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes,1 ,syn_type_float);
    unsigned runningMeanIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningMean, oneDimSizes,1, syn_type_float);
    unsigned runningVarIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningVar, oneDimSizes,1, syn_type_float);

    unsigned featureMapOut  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);
    unsigned featureMapOut2 = connectOutputTensorToInputTensor(featureMapOut);
    unsigned reluOutput     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);
    unsigned savedMeanOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1,  syn_type_float);
    unsigned iStdOut        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);

    addNodeToGraph("batch_norm_fwd_bf16", {featureMapIn, betaIn, gammaIn, runningMeanIn, runningVarIn},
            {featureMapOut, savedMeanOut, iStdOut}, &bnParams, sizeof(ns_BatchNormKernel::Params));
    addNodeToGraph("relu_fwd_bf16", {featureMapOut2}, {reluOutput});

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[reluOutput];
    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    float mean = 0;
    float iStd = 1;
    bfloat16 outRef[MAT_SIZE] = {};
    calcBatchNormForwardRef<bfloat16>(&mean, &iStd, &runningMean, &runningVar, outRef, 0.1, pFmInput, &beta, &gamma, fMsizes);
    for (unsigned i = 0; i < MAT_SIZE; i++)
    {
        outRef[i] = relu(outRef[i]);
    }
    validateResult(outRef, pFmOutput, MAT_SIZE);
}

TEST_F_GC(SynTrainingBNTest, batch_norm_add_relu_fusion_test_L2)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };
    float inValuesAdd[MAT_SIZE] = { 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0 };

    float runningMean = 0;
    float runningVar = 1;

    float beta = 0;
    float gamma = 1;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,syn_type_bf16);
    unsigned betaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes,1 ,syn_type_float);
    unsigned runningMeanIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningMean, oneDimSizes,1, syn_type_float);
    unsigned runningVarIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningVar, oneDimSizes,1, syn_type_float);

    unsigned featureMapOut  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);
    unsigned addIn1         = connectOutputTensorToInputTensor(featureMapOut);
    unsigned addIn2         = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValuesAdd, fMsizes, 4 ,syn_type_bf16);
    unsigned addOutput      = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);
    unsigned reluIn         = connectOutputTensorToInputTensor(addOutput);
    unsigned reluOutput     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);
    unsigned savedMeanOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1,  syn_type_float);
    unsigned iStdOut        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);

    addNodeToGraph("batch_norm_fwd_bf16", {featureMapIn, betaIn, gammaIn, runningMeanIn, runningVarIn},
            {featureMapOut, savedMeanOut, iStdOut}, &bnParams, sizeof(ns_BatchNormKernel::Params));
    addNodeToGraph("add_fwd_bf16",  {addIn2, addIn1}, {addOutput});
    addNodeToGraph("relu_fwd_bf16", {reluIn}, {reluOutput});

    compileTopology("gaudi_single_bn");
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[reluOutput];
    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    float mean = 0;
    float iStd = 1;
    bfloat16 outRef[MAT_SIZE] = {};
    calcBatchNormForwardRef<bfloat16>(&mean, &iStd, &runningMean, &runningVar, outRef, 0.1, pFmInput, &beta, &gamma, fMsizes);
    for (unsigned i = 0; i < MAT_SIZE; i++)
    {
        outRef[i] = relu((float)outRef[i] + 1);
    }
    validateResult(outRef, pFmOutput, MAT_SIZE);
}

TEST_F_GC(SynTrainingBNTest, bf16_triple_batch_norm_test)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };
    float runningMean = 0;
    float runningVar = 1;

    float beta = 0;
    float gamma = 1;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    unsigned featureMapIn1  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,syn_type_bf16);
    unsigned betaIn1        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn1       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes,1 ,syn_type_float);
    unsigned runningMeanIn1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningMean, oneDimSizes,1, syn_type_float);
    unsigned runningVarIn1  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningVar, oneDimSizes,1, syn_type_float);

    unsigned featureMapOut1  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);
    unsigned savedMeanOut1   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1,  syn_type_float);
    unsigned iStdOut1        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);


    unsigned featureMapIn2 = connectOutputTensorToInputTensor(featureMapOut1);
    unsigned betaIn2        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn2       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes,1 ,syn_type_float);
    unsigned runningMeanIn2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningMean, oneDimSizes,1, syn_type_float);
    unsigned runningVarIn2  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningVar, oneDimSizes,1, syn_type_float);

    unsigned savedMeanOut2   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1,  syn_type_float);
    unsigned iStdOut2        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);
    unsigned featureMapOut2  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);

    unsigned featureMapIn3 = connectOutputTensorToInputTensor(featureMapOut2);
    unsigned betaIn3        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn3       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes,1 ,syn_type_float);
    unsigned runningMeanIn3 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningMean, oneDimSizes,1, syn_type_float);
    unsigned runningVarIn3  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &runningVar, oneDimSizes,1, syn_type_float);

    unsigned savedMeanOut3   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1,  syn_type_float);
    unsigned iStdOut3        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);
    unsigned featureMapOut3  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);


    addNodeToGraph("batch_norm_fwd_bf16", {featureMapIn1, betaIn1, gammaIn1, runningMeanIn1, runningVarIn1},
            {featureMapOut1, savedMeanOut1, iStdOut1}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    addNodeToGraph("batch_norm_fwd_bf16", {featureMapIn2, betaIn2, gammaIn2, runningMeanIn2, runningVarIn2},
            {featureMapOut2, savedMeanOut2, iStdOut2}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    addNodeToGraph("batch_norm_fwd_bf16", {featureMapIn3, betaIn3, gammaIn3, runningMeanIn3, runningVarIn3},
            {featureMapOut3, savedMeanOut3, iStdOut3}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn1];
    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[featureMapOut3];
    float mean = 0;
    float iStd = 1;
    bfloat16 outRef[MAT_SIZE] = {};
    calcBatchNormForwardRef<bfloat16>(&mean, &iStd, &runningMean, &runningVar, outRef, 0.1, pFmInput, &beta, &gamma, fMsizes);
    validateResult(outRef, pFmOutput, MAT_SIZE);
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_conv_test_ASIC)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};
    TestSizes convWeightSizes = {channels, batch, 1, 1};

    synConvolutionParams convolutionParams;

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned betaIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data(), 1 ,syn_type_float);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned runningVarIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);

    unsigned featureMapOut  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, featureMapSizes.data(), 4, syn_type_bf16);
    unsigned savedMeanOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float);
    unsigned iStdOut        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned convInput      = connectOutputTensorToInputTensor(featureMapOut);
    unsigned convWeights    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, convWeightSizes.data(), 4 ,syn_type_bf16);
    unsigned convOut        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, featureMapSizes.data(), 4, syn_type_float);

    addNodeToGraph("batch_norm_fwd_bf16", {featureMapIn, betaIn, gammaIn, runningMeanIn, runningVarIn},
                  {featureMapOut, savedMeanOut, iStdOut}, &bnParams, sizeof(ns_BatchNormKernel::Params));
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {convInput, convWeights}, {convOut}, &convolutionParams,
                   sizeof(synConvolutionParams));

    compileTopology("gaudi_single_bn");
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    float* pBetaIn = (float*)m_hostBuffers[betaIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningVarIn = (float*)m_hostBuffers[runningVarIn];

    float* pConvOut = (float*)m_hostBuffers[convOut];
    bfloat16* bnRef = new bfloat16[sizeInElements];
    float* outRef = new float[sizeInElements];
    calcBatchNormForwardRef<bfloat16>(pRunningMeanIn, pRunningVarIn, pRunningMeanIn, pRunningVarIn,
                                      bnRef, bnParams.momentum, pFmInput, pBetaIn, pGammaIn, featureMapSizes.data());

    calculateFwdConvolution(m_tensorDescs[featureMapOut],
                            (char*)bnRef,
                            m_tensorDescs[convWeights],
                            (char*)m_hostBuffers[convWeights],
                            m_tensorDescs[convOut],
                            (char*)outRef,
                            convolutionParams,
                            m_deviceType);

    validateResult(outRef, pConvOut, sizeInElements);
    delete[] bnRef;
    delete[] outRef;
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_bwd_memcpy_test_ASIC)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned gradIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_bf16);
    unsigned gradOutIn = connectOutputTensorToInputTensor(gradOut);

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float);
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned gradOutCopy  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);

    addNodeToGraph("batch_norm_bwd_bf16", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {gradOutIn}, {gradOutCopy});

    compileTopology("gaudi_bn_bwd_memcopy");
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    bfloat16* pGradIn = (bfloat16*)m_hostBuffers[gradIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    bfloat16* pGradOut = (bfloat16*)m_hostBuffers[gradOut];
    bfloat16* pGradOutCopy = (bfloat16*)m_hostBuffers[gradOutCopy];

    bfloat16* gradOutputBufferRef = new bfloat16[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }
    calcBatchNormBackwardRef<bfloat16>(gradBetaRef,
                                       gradGammaRef,
                                       gradOutputBufferRef,
                                       pFmInput,
                                       pGradIn,
                                       pRunningMeanIn,
                                       pRunningIstdIn,
                                       pGammaIn,
                                       featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);
    for (unsigned i = 0; i < sizeInElements; i++)
    {
        ASSERT_EQ(pGradOutCopy[i].value(), pGradOut[i].value());
    }
    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, f32_batch_norm_add_relu_bwd_test_ASIC)
{
    ns_BatchNormKernel::ParamsV2 bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    bnParams.isTraining = true;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned addGradIn0     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);
    unsigned addGradIn1     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);
    unsigned addGradOut     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);

    unsigned reluGradIn     = connectOutputTensorToInputTensor(addGradOut);
    unsigned reluFwdOut     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);
    unsigned reluGradOut    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_float);

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);
    unsigned gradIn         = connectOutputTensorToInputTensor(reluGradOut);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_float);

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float);
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);

    addNodeToGraph("add_fwd_f32", {addGradIn0, addGradIn1},
                   {addGradOut}, nullptr);
    addNodeToGraph("relu_bwd_f32", {reluGradIn, reluFwdOut},
                   {reluGradOut}, nullptr);
    addNodeToGraph("batch_norm_bwd_f32", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    float* pFmInput = (float*)m_hostBuffers[featureMapIn];
    float* pGradIn = (float*)m_hostBuffers[reluGradOut];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    float* pGradOut = (float*)m_hostBuffers[gradOut];

    float* gradOutputBufferRef = new float[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }
    calcBatchNormBackwardRef<float>(gradBetaRef,
                                    gradGammaRef,
                                    gradOutputBufferRef,
                                    pFmInput,
                                    pGradIn,
                                    pRunningMeanIn,
                                    pRunningIstdIn,
                                    pGammaIn,
                                    featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_add_relu_bwd_test_ASIC)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned addGradIn0     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned addGradIn1     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned addGradOut     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);

    unsigned reluGradIn     = connectOutputTensorToInputTensor(addGradOut);
    unsigned reluFwdOut     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned reluGradOut    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, featureMapSizes.data(), 4, syn_type_bf16);

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned gradIn         = connectOutputTensorToInputTensor(reluGradOut);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_bf16);

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float);
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);

    addNodeToGraph("add_fwd_bf16", {addGradIn0, addGradIn1},
                   {addGradOut}, nullptr);
    addNodeToGraph("relu_bwd_bf16", {reluGradIn, reluFwdOut},
                   {reluGradOut}, nullptr);
    addNodeToGraph("batch_norm_bwd_bf16", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    bfloat16* pGradIn = (bfloat16*)m_hostBuffers[reluGradOut];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    bfloat16* gradOutputBufferRef = new bfloat16[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];
    bfloat16* pGradOut = (bfloat16*)m_hostBuffers[gradOut];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 1.0;
    }
    calcBatchNormBackwardRef<bfloat16>(gradBetaRef,
                                    gradGammaRef,
                                    gradOutputBufferRef,
                                    pFmInput,
                                    pGradIn,
                                    pRunningMeanIn,
                                    pRunningIstdIn,
                                    pGammaIn,
                                    featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, f32_batch_norm_relu_bwd_test_ASIC)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned reluGradIn     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);
    unsigned reluFwdOut     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);
    unsigned reluGradOut    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_float);

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float);
    unsigned gradIn         = connectOutputTensorToInputTensor(reluGradOut);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_float);

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float);
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);

    addNodeToGraph("relu_bwd_f32", {reluGradIn, reluFwdOut},
                   {reluGradOut}, nullptr);
    addNodeToGraph("batch_norm_bwd_f32", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    float* pFmInput = (float*)m_hostBuffers[featureMapIn];
    float* pGradIn = (float*)m_hostBuffers[reluGradOut];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    float* pGradOut = (float*)m_hostBuffers[gradOut];

    float* gradOutputBufferRef = new float[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }
    calcBatchNormBackwardRef<float>(gradBetaRef,
                                       gradGammaRef,
                                       gradOutputBufferRef,
                                       pFmInput,
                                       pGradIn,
                                       pRunningMeanIn,
                                       pRunningIstdIn,
                                       pGammaIn,
                                       featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_relu_bwd_test_ASIC)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned reluGradIn     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned reluFwdOut     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned reluGradOut    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_bf16);

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned gradIn         = connectOutputTensorToInputTensor(reluGradOut);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_bf16);

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float);
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);

    addNodeToGraph("relu_bwd_bf16", {reluGradIn, reluFwdOut},
                   {reluGradOut}, nullptr);
    addNodeToGraph("batch_norm_bwd_bf16", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    bfloat16* pGradIn = (bfloat16*)m_hostBuffers[reluGradOut];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    bfloat16* pGradOut = (bfloat16*)m_hostBuffers[gradOut];

    bfloat16* gradOutputBufferRef = new bfloat16[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }
    calcBatchNormBackwardRef<bfloat16>(gradBetaRef,
                                       gradGammaRef,
                                       gradOutputBufferRef,
                                       pFmInput,
                                       pGradIn,
                                       pRunningMeanIn,
                                       pRunningIstdIn,
                                       pGammaIn,
                                       featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, bf16_batch_norm_bwd_test_ASIC_CI)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned gradIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float);

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_bf16);

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float);
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float);

    addNodeToGraph("batch_norm_bwd_bf16", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology("gaudi_bn_bwd");
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    bfloat16* pGradIn = (bfloat16*)m_hostBuffers[gradIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    bfloat16* pGradOut = (bfloat16*)m_hostBuffers[gradOut];

    bfloat16* gradOutputBufferRef = new bfloat16[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }

    calcBatchNormBackwardRef<bfloat16>(gradBetaRef,
                                       gradGammaRef,
                                       gradOutputBufferRef,
                                       pFmInput,
                                       pGradIn,
                                       pRunningMeanIn,
                                       pRunningIstdIn,
                                       pGammaIn,
                                       featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, f32_batch_norm_bwd_test_ASIC_CI)
{
    ns_BatchNormKernel::Params bnParams;
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;
    unsigned channels = 3;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float, nullptr, "IFM");
    unsigned gradIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float, nullptr, "grad_in");
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float, nullptr, "running_mean");
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float, nullptr, "running_var");
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float, nullptr, "gamma_in");

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_float, nullptr, "grad_out");

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float, nullptr, "grad_beta");
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float, nullptr, "grad_gamma");

    addNodeToGraph("batch_norm_bwd_f32", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology("gaudi_bn_bwd");
    runTopology();

    float* pFmInput = (float*)m_hostBuffers[featureMapIn];
    float* pGradIn = (float*)m_hostBuffers[gradIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    float* pGradOut = (float*)m_hostBuffers[gradOut];

    float* gradOutputBufferRef = new float[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }
    calcBatchNormBackwardRef<float>(gradBetaRef,
                                    gradGammaRef,
                                    gradOutputBufferRef,
                                    pFmInput,
                                    pGradIn,
                                    pRunningMeanIn,
                                    pRunningIstdIn,
                                    pGammaIn,
                                    featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, f32_fused_batch_norm_grad_test_ASIC_CI)
{
    synTfBatchNormalizationParams bnParams;
    bnParams.variance_epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float, nullptr, "IFM");
    unsigned gradIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_float, nullptr, "grad_in");
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float, nullptr, "running_mean");
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float, nullptr, "running_var");
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float, nullptr, "gamma_in");

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_float, nullptr, "grad_out");

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float, nullptr, "grad_beta");
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float, nullptr, "grad_gamma");

    addNodeToGraph("tf_fused_batch_norm_grad", {gradIn, featureMapIn, gammaIn, runningMeanIn, runningIstdIn},
                   {gradOut, gradGamma, gradBeta}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology("gaudi_tf_fused_bn_bwd");
    runTopology();

    float* pFmInput = (float*)m_hostBuffers[featureMapIn];
    float* pGradIn = (float*)m_hostBuffers[gradIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    float* pGradOut = (float*)m_hostBuffers[gradOut];

    float* gradOutputBufferRef = new float[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }
    calcBatchNormBackwardRef<float>(gradBetaRef,
                                    gradGammaRef,
                                    gradOutputBufferRef,
                                    pFmInput,
                                    pGradIn,
                                    pRunningMeanIn,
                                    pRunningIstdIn,
                                    pGammaIn,
                                    featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

TEST_F_GC(SynTrainingBNTest, bf16_fused_batch_norm_grad_test_ASIC_CI)
{
    synTfBatchNormalizationParams bnParams;
    bnParams.variance_epsilon = 1e-05;
    unsigned channels = 64;
    unsigned height = 112;
    unsigned width = 112;
    unsigned batch = 64;
    unsigned sizeInElements = channels * height * width * batch;

    TestSizes featureMapSizes = {channels, height, width, batch};
    TestSizes channelSizes = {channels, 1, 1, 1};

    unsigned featureMapIn   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16, nullptr, "IFM");
    unsigned gradIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4 ,syn_type_bf16, nullptr, "grad_in");
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float, nullptr, "running_mean");
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float, nullptr, "running_var");
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, channelSizes.data() ,1, syn_type_float, nullptr, "gamma_in");

    unsigned gradOut   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, featureMapSizes.data(), 4, syn_type_bf16, nullptr, "grad_out");

    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1,  syn_type_float, nullptr, "grad_beta");
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, channelSizes.data(), 1, syn_type_float, nullptr, "grad_gamma");

    addNodeToGraph("tf_fused_batch_norm_grad", {gradIn, featureMapIn, gammaIn, runningMeanIn, runningIstdIn},
                   {gradOut, gradGamma, gradBeta}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology("gaudi_tf_fused_bn_bwd");
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];
    bfloat16* pGradIn = (bfloat16*)m_hostBuffers[gradIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    bfloat16* pGradOut = (bfloat16*)m_hostBuffers[gradOut];

    bfloat16* gradOutputBufferRef = new bfloat16[sizeInElements];
    float* gradBetaRef = new float[channels];
    float* gradGammaRef = new float[channels];

    for (int i = 0; i < channels; i++)
    {
        gradBetaRef[i] = 0.0;
        gradGammaRef[i] = 0.0;
    }
    calcBatchNormBackwardRef<bfloat16>(gradBetaRef,
                                    gradGammaRef,
                                    gradOutputBufferRef,
                                    pFmInput,
                                    pGradIn,
                                    pRunningMeanIn,
                                    pRunningIstdIn,
                                    pGammaIn,
                                    featureMapSizes.data());

    validateResult(gradOutputBufferRef, pGradOut, sizeInElements);

    delete[] gradOutputBufferRef;
    delete[] gradBetaRef;
    delete[] gradGammaRef;
}

template<typename DATA_TYPE>
void SynTrainingBNTest::moments_simple_test()
{
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    synDataType synDType = std::is_same<DATA_TYPE, float>::value ? syn_type_float : syn_type_bf16;

    /* inputs */
    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,synDType);

    /* outputs */
    unsigned meanOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);
    unsigned varOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1,  syn_type_float);

    addNodeToGraph("moments_fwd", {featureMapIn}, {meanOut, varOut}, nullptr);

    compileTopology();
    runTopology();

    float* pMeanOutData = (float*)m_hostBuffers[meanOut];
    float* pVarOutData = (float*)m_hostBuffers[varOut];
    DATA_TYPE* pInputData = (DATA_TYPE*)m_hostBuffers[featureMapIn];

    /* calc mean */
    float sum = 0;
    for (unsigned idx = 0; idx < MAT_SIZE; idx++)
    {
        sum += (float)pInputData[idx];
    }
    float ref_mean = sum / MAT_SIZE;
    validateResult(&ref_mean, pMeanOutData, 1);

    /* calc sigma sq */
    float ref_sigma_sq = 0;
    for (unsigned idx = 0; idx < MAT_SIZE; idx++)
    {
        ref_sigma_sq += pow((float)pInputData[idx] - ref_mean, 2);
    }
    float ref_var_out = ref_sigma_sq / MAT_SIZE;

    validateResult(&ref_var_out, pVarOutData, 1);
}

TEST_F_GC(SynTrainingBNTest, bf16_moments_simple_test)
{
    moments_simple_test<bfloat16>();
}

TEST_F_GC(SynTrainingBNTest, f32_moments_simple_test)
{
    moments_simple_test<float>();
}

template<typename DATA_TYPE>
void SynTrainingBNTest::moments_random_input_test()
{

    unsigned channels = 64;
    unsigned height   = 112;
    unsigned width    = 112;
    unsigned batch    = 64;

    unsigned fMsizes[4]      = {channels, height, width, batch};
    unsigned meanVarSizes[4] = {channels, 1, 1, 1};

    synDataType synDType = std::is_same<DATA_TYPE, float>::value ? syn_type_float : syn_type_bf16;

    /* inputs */
    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, fMsizes, 4 ,synDType,
                                                 nullptr, "IFM");

    /* outputs */
    unsigned meanOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, meanVarSizes, 1, syn_type_float,
                                           nullptr, "mean_out");
    unsigned varOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, meanVarSizes, 1,  syn_type_float,
                                           nullptr, "var_out");

    addNodeToGraph("moments_fwd", {featureMapIn}, {meanOut, varOut}, nullptr);

    compileTopology();
    runTopology();

    float* pMeanOutData = (float*)m_hostBuffers[meanOut];
    float* pVarOutData = (float*)m_hostBuffers[varOut];
    DATA_TYPE* pInputData = (DATA_TYPE*)m_hostBuffers[featureMapIn];

    /* calc mean */
    float* refMean = new float[channels]();
    float* refIstd = new float[channels]();
    float* beta    = new float[channels]();
    float* gamma   = new float[channels]();
    for (unsigned i = 0; i <channels; i++)
    {
        gamma[i] = 1.0;
    }

    calcBatchNormForwardRef<DATA_TYPE>(refMean, refIstd, nullptr, nullptr, nullptr, 0.1, pInputData, beta, gamma, fMsizes);

    float* refVar = new float[channels]();
    for (unsigned i = 0; i <channels; i++)
    {
        refVar[i] = 1.0 / pow(refIstd[i],2);
    }

    validateResult(refMean, pMeanOutData, channels, "mean");
    validateResult(refVar, pVarOutData, channels, "var");

    delete[] refMean;
    delete[] refIstd;
    delete[] beta;
    delete[] gamma;
    delete[] refVar;
}

/* TODO: SW-8984 - investigate this test failure */
TEST_F_GC(SynTrainingBNTest, bf16_moments_random_input_test_L2)
{
    moments_random_input_test<bfloat16>();
}

/* TODO: SW-8984 - investigate this tcest failure */
TEST_F_GC(SynTrainingBNTest, f32_moments_random_input_test_ASIC_CI)
{
    moments_random_input_test<float>();
}

TEST_F_GC(SynTrainingBNTest, bf16_tf_bn_no_moments)
{
    synTfBatchNormalizationParams bnParams;
    bnParams.variance_epsilon = 1e-05;
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };
    float mean = 1;
    float var = 2;

    float beta = 0;
    float gamma = 1;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,syn_type_bf16);
    unsigned betaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes, 1 ,syn_type_float);
    unsigned meanIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &mean, oneDimSizes, 1, syn_type_float);
    unsigned varIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &var, oneDimSizes, 1, syn_type_float);

    unsigned featureMapOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);

    addNodeToGraph("tf_batch_normalization_fwd", {featureMapIn, meanIn, varIn, betaIn, gammaIn}, {featureMapOut}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[featureMapOut];
    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];

    bfloat16 outRef[MAT_SIZE] = {};
    for(int i = 0; i < MAT_SIZE; i++)
    {
        outRef[i] = gamma * (((float)pFmInput[i] - mean) / sqrt(var + bnParams.variance_epsilon)) + beta;
    }

    validateResult(outRef, pFmOutput, MAT_SIZE);
}

TEST_F_GC(SynTrainingBNTest, bn_inf_bigc)
{
    ns_BatchNormKernel::Params kernelParams;
    memset(&kernelParams, 0, sizeof(kernelParams));
    kernelParams.epsilon     = 1e-05;

    const int c = 2048;
    const int w = 7;
    const int h = 7;
    const int n = 1;

    unsigned int ifmSizes[4]    = {c, w, h, n};
    unsigned int params_size[4] = {c, 1, 1, 1};

    const int ifmSize = 2048*7*7;
    float ifmInitializer[ifmSize] = {0};
    fillWithRandom<float>(ifmInitializer, ifmSize);

    const int bataSize = 2048;
    float betaInitializer[bataSize] = {0};
    fillWithRandom<float>(betaInitializer, bataSize);

    const int gammaSize = 2048;
    float gammaInitializer[gammaSize] = {0};
    fillWithRandom<float>(gammaInitializer, gammaSize);

    const int meanSize = 2048;
    float meanInitializer[gammaSize] = {0};
    fillWithRandom<float>(meanInitializer, meanSize);

    const int varSize = 2048;
    float varInitializer[varSize] = {0};
    fillWithRandom<float>(varInitializer, varSize);

    // Inputs
    unsigned ifm   = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmInitializer,   ifmSizes,    4 ,syn_type_bf16,  nullptr, "IFM");
    unsigned beta  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, betaInitializer,  params_size, 1, syn_type_float, nullptr, "betaInitializer");
    unsigned gamma = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, gammaInitializer, params_size, 1 ,syn_type_float, nullptr, "gamma");
    unsigned mean  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, meanInitializer,  params_size, 1, syn_type_float, nullptr, "mean");
    unsigned var   = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, varInitializer,   params_size, 1, syn_type_float, nullptr, "var");
    // Output
    unsigned ofm   = createPersistTensor(OUTPUT_TENSOR,     MEM_INIT_ALL_ZERO,         nullptr, ifmSizes, 4, syn_type_bf16, nullptr, "OFM");

    addNodeToGraph("batch_norm_inf_bf16",
                   {ifm, beta, gamma, mean, var},
                   {ofm},
                   &kernelParams, sizeof(ns_BatchNormKernel::Params));
    compileTopology();
    runTopology();

    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[ifm];
    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[ofm];

    bfloat16 outRef[ifmSize] = {};
    for(int i = 0; i < ifmSize; i++)
    {
        int paramsIndex = i % c;
        outRef[i]       = gammaInitializer[paramsIndex] * (((float)pFmInput[i] - meanInitializer[paramsIndex]) /
                                                     sqrt(varInitializer[paramsIndex] + kernelParams.epsilon)) +
                    betaInitializer[paramsIndex];
    }

    validateResult(outRef, pFmOutput, ifmSize);
}

TEST_F_GC(SynTrainingBNTest, f32_tf_bn_no_moments)
{
    synTfBatchNormalizationParams bnParams;
    bnParams.variance_epsilon = 1e-05;
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };
    float mean = 1;
    float var = 2;

    float beta = 1;
    float gamma = 2;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,syn_type_float);
    unsigned betaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes, 1 ,syn_type_float);
    unsigned meanIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &mean, oneDimSizes, 1, syn_type_float);
    unsigned varIn         = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &var, oneDimSizes, 1, syn_type_float);

    unsigned featureMapOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_float);

    addNodeToGraph("tf_batch_normalization_fwd", {featureMapIn, meanIn, varIn, betaIn, gammaIn}, {featureMapOut}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    float* pFmOutput = (float*)m_hostBuffers[featureMapOut];
    float* pFmInput = (float*)m_hostBuffers[featureMapIn];

    float outRef[MAT_SIZE] = {};
    for(int i = 0; i < MAT_SIZE; i++)
    {
        outRef[i] = gamma * ((pFmInput[i] - mean) / sqrt(var + bnParams.variance_epsilon)) + beta;
    }

    validateResult(outRef, pFmOutput, MAT_SIZE);
}

TEST_F_GC(SynTrainingBNTest, bf16_tf_bn_with_moments)
{
    synTfBatchNormalizationParams bnParams;
    bnParams.variance_epsilon = 1e-05;
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };

    float beta = 0;
    float gamma = 1;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,syn_type_bf16);
    unsigned betaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes, 1 ,syn_type_float);
    unsigned meanOut       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);
    unsigned varOut        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, oneDimSizes, 1, syn_type_float);
    unsigned meanIn        = connectOutputTensorToInputTensor(meanOut);
    unsigned varIn         = connectOutputTensorToInputTensor(varOut);

    unsigned featureMapOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_bf16);

    addNodeToGraph("moments_fwd", {featureMapIn}, {meanOut, varOut}, nullptr);
    addNodeToGraph("tf_batch_normalization_fwd", {featureMapIn, meanIn, varIn, betaIn, gammaIn}, {featureMapOut}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[featureMapOut];
    float* pMean = (float*)m_hostBuffers[meanOut];
    float* pVar = (float*)m_hostBuffers[varOut];
    bfloat16* pFmInput = (bfloat16*)m_hostBuffers[featureMapIn];

    bfloat16 outRef[MAT_SIZE] = {};
    for(int i = 0; i < MAT_SIZE; i++)
    {
        outRef[i] = gamma * (((float)pFmInput[i] - *pMean) / sqrt(*pVar + bnParams.variance_epsilon)) + beta;
    }

    validateResult(outRef, pFmOutput, MAT_SIZE);
}

TEST_F_GC(SynTrainingBNTest, f32_tf_bn_with_moments)
{
    synTfBatchNormalizationParams bnParams;
    bnParams.variance_epsilon = 1e-05;
    // a simple test a single bn_fwd with mean 0 and var 2.
    float inValues[MAT_SIZE] = { -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0,
                                 -2.0, -1.0, 0.0, 1.0, 2.0 };

    float beta = 1;
    float gamma = 2;

    unsigned fMsizes[4] = {1, 5, 5, 1};
    unsigned oneDimSizes[4] = {1, 1, 1, 1};

    unsigned featureMapIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, fMsizes, 4 ,syn_type_float);
    unsigned betaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &beta, oneDimSizes, 1,syn_type_float);
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &gamma, oneDimSizes,1 ,syn_type_float);
    unsigned meanOut       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes, 1, syn_type_float);
    unsigned varOut        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, oneDimSizes, 1, syn_type_float);
    unsigned meanIn        = connectOutputTensorToInputTensor(meanOut);
    unsigned varIn         = connectOutputTensorToInputTensor(varOut);

    unsigned featureMapOut  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fMsizes, 4, syn_type_float);

    addNodeToGraph("moments_fwd", {featureMapIn}, {meanOut, varOut}, nullptr);
    addNodeToGraph("tf_batch_normalization_fwd", {featureMapIn, meanIn, varIn, betaIn, gammaIn}, {featureMapOut}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    runTopology();

    float* pFmOutput = (float*)m_hostBuffers[featureMapOut];
    float* pMean = (float*)m_hostBuffers[meanOut];
    float* pVar = (float*)m_hostBuffers[varOut];
    float* pFmInput = (float*)m_hostBuffers[featureMapIn];

    float outRef[MAT_SIZE] = {};
    for(int i = 0; i < MAT_SIZE; i++)
    {
        outRef[i] = gamma * ((pFmInput[i] - *pMean) / sqrt(*pVar + bnParams.variance_epsilon)) + beta;
    }

    validateResult(outRef, pFmOutput, MAT_SIZE);
}

class BnGemmBnTest
: public testing::WithParamInterface<std::tuple<uint32_t, uint32_t, uint32_t>>
, public SynTrainingTestInfra
{
public:
    static void SetUpTestSuite() { ReleaseDevice(); }
};

TEST_P_GC(BnGemmBnTest, full_bn_gemm_bn_ASIC_CI)
{
    uint32_t bhw = std::get<0>(GetParam());
    uint32_t inChannels = std::get<1>(GetParam());
    uint32_t outChannels = std::get<2>(GetParam());UNUSED(outChannels);

    unsigned inFMSizes[]   = {inChannels, bhw, 1, 1};
    unsigned inChannelsSizes[]  = {inChannels};
    unsigned wghSizes[] = {outChannels, inChannels, 1, 1};
    unsigned outFMSizes[] = {outChannels, bhw, 1, 1};
    unsigned outChannelsSizes[] = {outChannels};

    // Add 1st BN
    unsigned featureMapIn0   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inFMSizes, ARRAY_SIZE(inFMSizes), syn_type_float, nullptr, "featureMapIn0");
    unsigned betaIn0         = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inChannelsSizes, ARRAY_SIZE(inChannelsSizes), syn_type_float, nullptr, "betaIn0");
    unsigned gammaIn0        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, inChannelsSizes, ARRAY_SIZE(inChannelsSizes) ,syn_type_float, nullptr, "gammaIn0");
    unsigned runningMeanIn0  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inChannelsSizes, ARRAY_SIZE(inChannelsSizes), syn_type_float, nullptr, "runningMeanIn0");
    unsigned runningVarIn0   = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, inChannelsSizes, ARRAY_SIZE(inChannelsSizes), syn_type_float, nullptr, "runningVarIn0");

    unsigned featureMapOut0  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inFMSizes, ARRAY_SIZE(inFMSizes), syn_type_float, nullptr, "featureMapOut0");
    unsigned savedMeanOut0   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inChannelsSizes, ARRAY_SIZE(inChannelsSizes),  syn_type_float, nullptr, "savedMeanOut0");
    unsigned iStdOut0        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inChannelsSizes, ARRAY_SIZE(inChannelsSizes), syn_type_float, nullptr, "iStdOut0");

    ns_BatchNormKernel::Params bnParams0;
    bnParams0.momentum = 0.1;
    bnParams0.threshold.f = 1e-05;
    bnParams0.epsilon = 1e-05;
    addNodeToGraph("batch_norm_fwd_f32", {featureMapIn0, betaIn0, gammaIn0, runningMeanIn0, runningVarIn0}, {featureMapOut0, savedMeanOut0, iStdOut0}, &bnParams0, sizeof bnParams0, "1stBN");

    // Add GEMM
    unsigned gemmIn = connectOutputTensorToInputTensor(featureMapOut0);
    unsigned gemmWgh = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wghSizes, ARRAY_SIZE(wghSizes), syn_type_float, nullptr, "WGH");

    unsigned gemmOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outFMSizes, ARRAY_SIZE(outFMSizes), syn_type_float, nullptr, "GemmOut");

    synGEMMParams gemmParams;
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {gemmIn, gemmWgh}, {gemmOut}, &gemmParams, sizeof gemmParams, "GEMM");

    // Add 2nd BN
    unsigned featureMapIn1   = connectOutputTensorToInputTensor(gemmOut);
    unsigned betaIn1         = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outChannelsSizes, ARRAY_SIZE(outChannelsSizes), syn_type_float, nullptr, "betaIn1");
    unsigned gammaIn1        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, outChannelsSizes, ARRAY_SIZE(outChannelsSizes) ,syn_type_float, nullptr, "gammaIn1");
    unsigned runningMeanIn1  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outChannelsSizes, ARRAY_SIZE(outChannelsSizes), syn_type_float, nullptr, "runningMeanIn1");
    unsigned runningVarIn1   = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, outChannelsSizes, ARRAY_SIZE(outChannelsSizes), syn_type_float, nullptr, "runningVarIn1");

    unsigned featureMapOut1  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outFMSizes, ARRAY_SIZE(outFMSizes), syn_type_float, nullptr, "featureMapOut1");
    unsigned savedMeanOut1   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outChannelsSizes, ARRAY_SIZE(outChannelsSizes),  syn_type_float, nullptr, "savedMeanOut1");
    unsigned iStdOut1        = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outChannelsSizes, ARRAY_SIZE(outChannelsSizes), syn_type_float, nullptr, "iStdOut1");

    ns_BatchNormKernel::Params bnParams1;
    bnParams1.momentum = 0.1;
    bnParams1.threshold.f = 1e-05;
    bnParams1.epsilon = 1e-05;
    addNodeToGraph("batch_norm_fwd_f32", {featureMapIn1, betaIn1, gammaIn1, runningMeanIn1, runningVarIn1}, {featureMapOut1, savedMeanOut1, iStdOut1}, &bnParams1, sizeof bnParams1, "2ndBN");

    compileAndRun();

    float* pFmInput = (float*)m_hostBuffers[featureMapIn0];
    float* beta = (float*)m_hostBuffers[betaIn0];
    float* gamma = (float*)m_hostBuffers[gammaIn0];
    float* mean = new float[inChannels];
    float* var = new float[inChannels];
    float* outRef = new float[inChannels * bhw];
    calcBatchNormForwardRef<float>(mean, var, nullptr, nullptr, outRef, 0.1, pFmInput, beta, gamma, inFMSizes);
    float* pFmOutput = (float*)m_hostBuffers[featureMapOut0];
    validateResult(outRef, pFmOutput, bhw*inChannels);
    delete[] mean;
    delete[] var;
    delete[] outRef;

    CoordArray outIdx;
    synConvolutionParams gemmBy1x1ConvParams{};
    ASSERT_TRUE(checkFwdConvolution(m_tensorDescs[featureMapOut0],
                                    (char*)pFmOutput,
                                    m_tensorDescs[gemmWgh],
                                    (char*)m_hostBuffers[gemmWgh],
                                    m_tensorDescs[gemmOut],
                                    (char*)m_hostBuffers[gemmOut],
                                    gemmBy1x1ConvParams,
                                    outIdx,
                                    m_deviceType));

    pFmInput = (float*)m_hostBuffers[gemmOut];
    beta = (float*)m_hostBuffers[betaIn1];
    gamma = (float*)m_hostBuffers[gammaIn1];
    mean = new float[outChannels];
    var = new float[outChannels];
    outRef = new float[outChannels * bhw];
    calcBatchNormForwardRef<float>(mean, var, nullptr, nullptr, outRef, 0.1, pFmInput, beta, gamma, outFMSizes);
    pFmOutput = (float*)m_hostBuffers[featureMapOut1];
    validateResult(outRef, pFmOutput, bhw*outChannels);
    delete[] mean;
    delete[] var;
    delete[] outRef;
}

INSTANTIATE_TEST_SUITE_P(SynTrainingBundleTests,
                         BnGemmBnTest,
                         ::testing::Combine(::testing::Values<uint32_t>(64, 256, 1024, 4096),      // bhw
                                            ::testing::Values<uint32_t>(16, 64, 256, 1024, 2048),  // inChannels
                                            ::testing::Values<uint32_t>(128, 512)));

TEST_F_GC(SynTrainingBNTest, batch_norm_stage1)
{
    GlobalConfTestSetter setter("MAX_RMW_TENSOR_BYTES", "8192");

    uint32_t channels = 512;
    TestSizes inSizes = {channels, 64, 64, 1, 1};
    TestSizes runningMeanSizes = {channels, 1, 1, 1, 1};
    TestSizes outSize          = {channels, 3};

    // Add 1st BN
    unsigned input           = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSizes.data(), 4, syn_type_float, nullptr, "input");
    unsigned runningMean     = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, runningMeanSizes.data(), 1, syn_type_float, nullptr, "runningMean");
    unsigned output          = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, outSize.data(), 2 ,syn_type_float, nullptr, "output");

    ns_BatchNormStage1Kernel::Params stage1Params;
    stage1Params.disable_beta_gamma_update = 0;
    stage1Params.N = 64*64;

    addNodeToGraph("batch_norm_stage1_fwd_f32", {input, runningMean}, {output}, &stage1Params, sizeof(stage1Params), "batch_norm_stage1_fwd_f32");

    compileAndRun();

    float* inputData = (float*)m_hostBuffers[input];
    float* outputData = (float*)m_hostBuffers[output];

    uint32_t inputSpatialSize = multiplyElements(std::next(inSizes.begin()), inSizes.end());
    float    epsilon          = 0.0015f;
    for (uint32_t c = 0; c < channels; ++c)
    {
        float sum = 0;
        float sum_square = 0;
        for (uint32_t e = 0; e < inputSpatialSize; ++e)
        {
            sum += inputData[e * channels + c];
            sum_square += inputData[e * channels + c] * inputData[e * channels + c];
        }

        ASSERT_TRUE(float_eq(sum, outputData[c], epsilon))
            << "Wrong value at index " << c << ", Exp: " << sum << ", Got " << outputData[c];
        ASSERT_TRUE(float_eq(sum_square, outputData[c + channels], epsilon))
            << "Wrong value at index " << c + channels << ", Exp: " << sum_square << ", Got "
            << outputData[c + channels];
    }
}
