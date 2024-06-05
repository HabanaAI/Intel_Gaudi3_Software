#include "gc_dynamic_shapes_infra.h"
#include "synapse_common_types.h"
#include "tpc_batch_norm_test.h"

class SynGaudiDynamicBatchNormTest
: public SynGaudiDynamicShapesTestsInfra
, public ::testing::WithParamInterface<std::tuple<size_t /*ActualW*/, size_t /*ActualH*/, size_t /*ActualB*/>>
{
};

class SynGaudiTypedDynamicBnTest : public SynGaudiDynamicBatchNormTest
{
public:
    template<typename valType>
    void fwdTest(size_t height, size_t width, size_t batch);

    template<typename valType>
    void convProducerFwdTest(size_t height, size_t width, size_t batch);

    template<typename valType>
    void convConsumerFwdTest(size_t height, size_t width, size_t batch);

    template<typename valType>
    void bwdTest(size_t height, size_t width, size_t batch);

    template<typename valType>
    void convProducerBwdTest(size_t height, size_t width, size_t batch);

    template<typename valType>
    void convConsumerBwdTest(size_t height, size_t width, size_t batch);

    static const size_t C = 16;
    static const size_t MAX_W = 128;
    static const size_t MIN_W = 64;
    static const size_t MAX_H = 128;
    static const size_t MIN_H = 64;
    static const size_t MAX_B = 32;
    static const size_t MIN_B = 2;
    static const size_t BATCH = 16;
    static const size_t TENSOR_DIMS = 4;

private:
    template<typename valType>
    std::string getFwdGuid()
    {
        synDataType synType = dataTypeToSynType<valType>();

        if (synType == syn_type_float)
        {
            return "batch_norm_fwd_f32";
        }
        else if (synType == syn_type_bf16)
        {
            return "batch_norm_fwd_bf16";
        }
        else
        {
            return "";
        }
    }
    template<typename valType>
    std::string getBwdGuid()
    {
        synDataType synType = dataTypeToSynType<valType>();

        if (synType == syn_type_float)
        {
            return "batch_norm_bwd_f32";
        }
        else if (synType == syn_type_bf16)
        {
            return "batch_norm_bwd_bf16";
        }
        else
        {
            return "";
        }
    }
};

class SynGaudiDynamicChannelsBatchNormTest
: public SynGaudiDynamicShapesTestsInfra
, public ::testing::WithParamInterface<unsigned /* ActualC */>
{
public:

    static const size_t WIDTH = 128;
    static const size_t HEIGHT = 128;
    static const size_t BATCH = 16;
    static const size_t TENSOR_DIMS = 4;
    static const size_t MAX_C = 16;
    static const size_t MIN_C = 2;

};

template<typename valType>
void SynGaudiTypedDynamicBnTest::fwdTest(size_t height, size_t width, size_t batch)
{
    synDataType synType     = dataTypeToSynType<valType>();
    std::string bn_fwd_guid = getFwdGuid<valType>();
    ASSERT_TRUE(bn_fwd_guid.size() > 0) << "invalid data type";

    size_t inActualW = height;
    size_t inActualH = width;
    size_t inActualB = batch;

    unsigned inMaxSizes[] = {C, MAX_W, MAX_H, MAX_B};
    unsigned inMinSizes[] = {C, MIN_W, MIN_H, MIN_B};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            TENSOR_DIMS,
                                            synType,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned bnTensorSizes[] = {C};

    unsigned bnBetaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                bnTensorSizes, 1, syn_type_float, nullptr, "Beta");
    unsigned bnGammaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                 bnTensorSizes, 1, syn_type_float, nullptr, "Gamma");
    unsigned bnInMeanTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                  bnTensorSizes, 1, syn_type_float, nullptr, "In Mean");
    unsigned bnInVarTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr,
                                                 bnTensorSizes, 1, syn_type_float, nullptr, "In Var");

    unsigned bnOutMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr,
                                            bnTensorSizes, 1, syn_type_float);
    unsigned bnOutStdTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr,
                                           bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr,
                                               bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunVarTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr,
                                              bnTensorSizes, 1, syn_type_float);

    ns_BatchNormKernel::Params bnParams {};
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             inMaxSizes,
                                             TENSOR_DIMS,
                                             synType,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inMinSizes);

    addNodeToGraph(bn_fwd_guid.c_str(),
                   {inTensor, bnBetaTensor, bnGammaTensor, bnInMeanTensor, bnInVarTensor},
                   {outTensor, bnOutMeanTensor, bnOutStdTensor, bnOutRunMeanTensor, bnOutRunVarTensor},
                   &bnParams,
                   sizeof bnParams);

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    unsigned inActualSizes[] = {C, inActualW, inActualH, inActualB};
    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, inActualSizes);

    auto* inData = castHostInBuffer<valType>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    auto                 totalActualSize = multiplyElements(inActualSizes, inActualSizes + TENSOR_DIMS);
    std::array<float, C> mean{};
    std::array<float, C> runningMean{};
    mean.fill(0.f);
    runningMean.fill(0.f);
    std::array<float, C> iStd{}, runningVar{};
    iStd.fill(1.f);
    runningVar.fill(1.f);
    std::vector<valType> outputRef(totalActualSize);

    calcBatchNormForwardRef<valType>(mean.data(),
                                     iStd.data(),
                                     runningMean.data(),
                                     runningVar.data(),
                                     outputRef.data(),
                                     bnParams.momentum,
                                     inData,
                                     castHostInBuffer<float>(bnBetaTensor),
                                     castHostInBuffer<float>(bnGammaTensor),
                                     inActualSizes);

    auto* output = castHostOutBuffer<valType>(outTensor);

    validateResult(outputRef.data(), output, totalActualSize);
}

template<typename valType>
void SynGaudiTypedDynamicBnTest::convProducerFwdTest(size_t height, size_t width, size_t batch)
{
    synDataType synType     = dataTypeToSynType<valType>();
    std::string bn_fwd_guid = getFwdGuid<valType>();
    ASSERT_TRUE(bn_fwd_guid.size() > 0) << "invalid data type";

    size_t inActualW = width;
    size_t inActualH = height;
    size_t inActualB = batch;
    // Create the static part of the graph that is inserted as weights to the conv.
    unsigned inMaxSizes[]  = {C, MAX_W, MAX_H, MAX_B};
    unsigned inMinSizes[]  = {C, MIN_W, MIN_H, MIN_B};
    unsigned weightSizes[] = {C, C, 1, 1};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            TENSOR_DIMS,
                                            synType,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned weightTensor =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, weightSizes, TENSOR_DIMS, synType);

    synConvolutionParams params;

    unsigned bnTensorSizes[] = {C};

    unsigned bnBetaTensor   = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                bnTensorSizes,
                                                1,
                                                syn_type_float,
                                                nullptr,
                                                "Beta");
    unsigned bnGammaTensor  = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "Gamma");
    unsigned bnInMeanTensor = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  bnTensorSizes,
                                                  1,
                                                  syn_type_float,
                                                  nullptr,
                                                  "In Mean");
    unsigned bnInVarTensor  = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_ALL_ONES,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "In Var");

    unsigned bnOutMeanTensor    = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);
    unsigned bnOutStdTensor     = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunVarTensor  = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);

    ns_BatchNormKernel::Params bnParams {};
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon     = 1e-05;

    unsigned bnOutTensor = createTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        inMaxSizes,
                                        TENSOR_DIMS,
                                        synType,
                                        nullptr,
                                        inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             inMaxSizes,
                                             TENSOR_DIMS,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inMinSizes);

    addNodeToGraph(bn_fwd_guid.c_str(),
                   {inTensor, bnBetaTensor, bnGammaTensor, bnInMeanTensor, bnInVarTensor},
                   {bnOutTensor, bnOutMeanTensor, bnOutStdTensor, bnOutRunMeanTensor, bnOutRunVarTensor},
                   &bnParams,
                   sizeof bnParams);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {bnOutTensor, weightTensor},
                   {outTensor},
                   &params,
                   sizeof(params));
    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    unsigned inActualSizes[] = {C, inActualW, inActualH, inActualB};
    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, inActualSizes);

    auto* inData = castHostInBuffer<valType>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + TENSOR_DIMS);

    std::array<float, C> mean {};
    std::array<float, C> runningMean {};
    mean.fill(0.f);
    runningMean.fill(0.f);
    std::array<float, C> iStd {}, runningVar {};
    iStd.fill(1.f);
    runningVar.fill(1.f);
    std::vector<valType> outputBnRef(totalActualSize);
    std::fill(outputBnRef.begin(), outputBnRef.end(), 0.0f);

    valType*           weightsData = castHostInBuffer<valType>(weightTensor);
    std::vector<float> outConv(totalActualSize);

    auto inDesc      = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    auto weightDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(weightTensor));
    auto convOutDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(outTensor));

    memcpy(inDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);
    memcpy(convOutDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);

    calcBatchNormForwardRef<valType>(mean.data(),
                                     iStd.data(),
                                     runningMean.data(),
                                     runningVar.data(),
                                     outputBnRef.data(),
                                     bnParams.momentum,
                                     inData,
                                     castHostInBuffer<float>(bnBetaTensor),
                                     castHostInBuffer<float>(bnGammaTensor),
                                     inActualSizes);

    calculateFwdConvolution(inDesc,
                            (char*)outputBnRef.data(),
                            weightDesc,
                            (char*)weightsData,
                            convOutDesc,
                            (char*)outConv.data(),
                            params,
                            m_deviceType);
    auto* output = castHostOutBuffer<float>(outTensor);
    validateResult(outConv.data(), output, totalActualSize);
}

template<typename valType>
void SynGaudiTypedDynamicBnTest::convConsumerFwdTest(size_t height, size_t width, size_t batch)
{
    synDataType synType     = dataTypeToSynType<valType>();
    std::string bn_fwd_guid = getFwdGuid<valType>();
    ASSERT_TRUE(bn_fwd_guid.size() > 0) << "invalid data type";

    size_t inActualW = width;
    size_t inActualH = height;
    size_t inActualB = batch;
    // Create the static part of the graph that is inserted as weights to the conv.
    unsigned inMaxSizes[]  = {C, MAX_W, MAX_H, MAX_B};
    unsigned inMinSizes[]  = {C, MIN_W, MIN_H, MIN_B};

    unsigned inActualSizes[] = {C, inActualW, inActualH, inActualB};

    unsigned weightSizes[] = {C, C, 1, 1};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            TENSOR_DIMS,
                                            synType,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned weightTensor =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, weightSizes, TENSOR_DIMS, synType);

    unsigned convOutTensor = createTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          inMaxSizes,
                                          TENSOR_DIMS,
                                          synType,
                                          nullptr,
                                          inMinSizes);

    synConvolutionParams params;

    unsigned bnTensorSizes[] = {C};

    unsigned bnBetaTensor   = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                bnTensorSizes,
                                                1,
                                                syn_type_float,
                                                nullptr,
                                                "Beta");
    unsigned bnGammaTensor  = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "Gamma");
    unsigned bnInMeanTensor = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  bnTensorSizes,
                                                  1,
                                                  syn_type_float,
                                                  nullptr,
                                                  "In Mean");
    unsigned bnInVarTensor  = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_ALL_ONES,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "In Var");

    unsigned bnOutMeanTensor    = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);
    unsigned bnOutStdTensor     = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);
    unsigned bnOutRunVarTensor  = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, bnTensorSizes, 1, syn_type_float);

    ns_BatchNormKernel::Params bnParams {};
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon     = 1e-05;

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             inMaxSizes,
                                             TENSOR_DIMS,
                                             synType,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inMinSizes);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {inTensor, weightTensor},
                   {convOutTensor},
                   &params,
                   sizeof(params));

    addNodeToGraph(bn_fwd_guid.c_str(),
                   {convOutTensor, bnBetaTensor, bnGammaTensor, bnInMeanTensor, bnInVarTensor},
                   {outTensor, bnOutMeanTensor, bnOutStdTensor, bnOutRunMeanTensor, bnOutRunVarTensor},
                   &bnParams,
                   sizeof bnParams);

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, inActualSizes);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + TENSOR_DIMS);

    std::array<float, C> mean {};
    std::array<float, C> runningMean {};
    mean.fill(0.f);
    runningMean.fill(0.f);
    std::array<float, C> iStd {}, runningVar {};
    iStd.fill(1.f);
    runningVar.fill(1.f);
    std::vector<float> outputBnRef(totalActualSize);
    std::fill(outputBnRef.begin(), outputBnRef.end(), 0.0f);
    auto inDesc      = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    auto weightDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(weightTensor));
    auto convOutDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(outTensor));
    // because CPU calculator supports only floats, we calculate refference in float.
    float*             pConvRefferenceIn     = nullptr;
    float*             pConvRefferenceWeight = nullptr;
    std::vector<float> convRefferenceIn;
    std::vector<float> convRefferenceWeight;
    std::vector<float> outputConv(totalActualSize);

    if (synType == syn_type_float)
    {
        pConvRefferenceIn     = castHostInBuffer<float>(inTensor);
        pConvRefferenceWeight = castHostInBuffer<float>(weightTensor);
    }
    else if (synType == syn_type_bf16)
    {
        convRefferenceIn.reserve(totalActualSize);
        convRefferenceWeight.reserve(C * C);
        bfloat16* convInData = castHostInBuffer<bfloat16>(inTensor);
        bfloat16* weightData = castHostInBuffer<bfloat16>(weightTensor);

        for (unsigned i = 0; i < totalActualSize; i++)
        {
            convRefferenceIn[i] = (float)convInData[i];
        }
        for (unsigned i = 0; i < C * C; i++)
        {
            convRefferenceWeight[i] = (float)weightData[i];
        }
        pConvRefferenceIn     = convRefferenceIn.data();
        pConvRefferenceWeight = convRefferenceWeight.data();
    }
    else
    {
        ASSERT_ANY_THROW() << "invalid data type";
    }
    inDesc.m_dataType      = syn_type_float;
    weightDesc.m_dataType  = syn_type_float;
    convOutDesc.m_dataType = syn_type_float;

    memcpy(inDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);
    memcpy(convOutDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);

    calculateFwdConvolution(inDesc,
                            (char*)pConvRefferenceIn,
                            weightDesc,
                            (char*)pConvRefferenceWeight,
                            convOutDesc,
                            (char*)outputConv.data(),
                            params,
                            m_deviceType);

    calcBatchNormForwardRef<float>(mean.data(),
                                   iStd.data(),
                                   runningMean.data(),
                                   runningVar.data(),
                                   outputBnRef.data(),
                                   bnParams.momentum,
                                   outputConv.data(),
                                   castHostInBuffer<float>(bnBetaTensor),
                                   castHostInBuffer<float>(bnGammaTensor),
                                   inActualSizes);

    auto* output = castHostOutBuffer<valType>(outTensor);
    validateResult(outputBnRef.data(), output, totalActualSize);
}

template<typename valType>
void SynGaudiTypedDynamicBnTest::bwdTest(size_t height, size_t width, size_t batch)
{
    size_t inActualW      = width;
    size_t inActualH      = height;
    size_t inActualB      = batch;
    size_t sizeInElements = C * inActualW * inActualH * inActualB;

    synDataType synType     = dataTypeToSynType<valType>();
    std::string bn_bwd_guid = getBwdGuid<valType>();
    ASSERT_TRUE(bn_bwd_guid.size() > 0) << "invalid data type";

    unsigned inMaxSizes[] = {C, MAX_W, MAX_H, MAX_B};
    unsigned inMinSizes[] = {C, MIN_W, MIN_H, MIN_B};
    unsigned bnTensorSizes[] = {C};
    ns_BatchNormKernel::Params bnParams{};
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;

    unsigned featureMapIn = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                inMaxSizes,
                                                TENSOR_DIMS,
                                                synType,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                inMinSizes);
    unsigned gradIn       = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          inMaxSizes,
                                          TENSOR_DIMS,
                                          synType,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          inMinSizes);
    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, bnTensorSizes, 1, syn_type_float, nullptr, "running_mean");
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, bnTensorSizes ,1, syn_type_float, nullptr, "running_var");
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, bnTensorSizes ,1, syn_type_float, nullptr, "gamma_in");



    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, bnTensorSizes, 1,  syn_type_float, nullptr, "grad_beta");
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, bnTensorSizes, 1, syn_type_float, nullptr, "grad_gamma");

    unsigned gradOut = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           inMaxSizes,
                                           TENSOR_DIMS,
                                           synType,
                                           nullptr,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           inMinSizes);

    addNodeToGraph(bn_bwd_guid.c_str(),
                   {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma},
                   &bnParams,
                   sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";


    unsigned inActualSizes[] = {C, inActualW, inActualH, inActualB};
    setActualSizes(featureMapIn, inActualSizes);
    setActualSizes(gradOut, inActualSizes);
    setActualSizes(gradIn, inActualSizes);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    valType* pFmInput       = (valType*) m_hostBuffers[featureMapIn];
    valType* pGradIn        = (valType*) m_hostBuffers[gradIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    valType* pGradOut = (valType*) m_hostBuffers[gradOut];

    float* pGradBetaOut = (float*)m_hostBuffers[gradBeta];
    float* pGradGammaOut = (float*)m_hostBuffers[gradGamma];

    std::vector<valType> gradOutputBufferRef(sizeInElements);
    std::array<float, C> gradBetaRef{};
    std::array<float, C> gradGammaRef{};
    gradBetaRef.fill(0.f);
    gradGammaRef.fill(0.f);

    calcBatchNormBackwardRef<valType>(gradBetaRef.data(),
                                      gradGammaRef.data(),
                                      gradOutputBufferRef.data(),
                                      pFmInput,
                                      pGradIn,
                                      pRunningMeanIn,
                                      pRunningIstdIn,
                                      pGammaIn,
                                      inActualSizes);

    validateResult(gradOutputBufferRef.data(), pGradOut, sizeInElements);
    validateResult(gradBetaRef.data(), pGradBetaOut, C);
    validateResult(gradGammaRef.data(), pGradGammaOut, C);

}

template<typename valType>
void SynGaudiTypedDynamicBnTest::convProducerBwdTest(size_t height, size_t width, size_t batch)
{
    size_t inActualW      = width;
    size_t inActualH      = height;
    size_t inActualB      = batch;
    size_t sizeInElements = C * inActualW * inActualH * inActualB;

    synDataType synType     = dataTypeToSynType<valType>();
    std::string bn_bwd_guid = getBwdGuid<valType>();
    ASSERT_TRUE(bn_bwd_guid.size() > 0) << "invalid data type";

    unsigned inMaxSizes[]    = {C, MAX_W, MAX_H, MAX_B};
    unsigned inMinSizes[]    = {C, MIN_W, MIN_H, MIN_B};
    unsigned bnTensorSizes[] = {C};
    unsigned weightSizes[]   = {C, C, 1, 1};

    ns_BatchNormKernel::Params bnParams {};
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon     = 1e-05;

    synConvolutionParams params;

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            TENSOR_DIMS,
                                            synType,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned weightTensor =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, weightSizes, TENSOR_DIMS, synType);

    unsigned gradIn        = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          inMaxSizes,
                                          TENSOR_DIMS,
                                          synType,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          inMinSizes);
    unsigned runningMeanIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "running_mean");
    unsigned runningIstdIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_ALL_ONES,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "running_var");
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_ALL_ONES,
                                           nullptr,
                                           bnTensorSizes,
                                           1,
                                           syn_type_float,
                                           nullptr,
                                           "gamma_in");

    unsigned gradBeta  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            bnTensorSizes,
                                            1,
                                            syn_type_float,
                                            nullptr,
                                            "grad_beta");
    unsigned gradGamma = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             bnTensorSizes,
                                             1,
                                             syn_type_float,
                                             nullptr,
                                             "grad_gamma");

    unsigned bnOutTensor = createTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        inMaxSizes,
                                        TENSOR_DIMS,
                                        synType,
                                        nullptr,
                                        inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             inMaxSizes,
                                             TENSOR_DIMS,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inMinSizes);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {bnOutTensor, weightTensor},
                   {outTensor},
                   &params,
                   sizeof(params));

    addNodeToGraph(bn_bwd_guid.c_str(),
                   {inTensor, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {bnOutTensor, gradBeta, gradGamma},
                   &bnParams,
                   sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    unsigned inActualSizes[] = {C, inActualW, inActualH, inActualB};
    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, inActualSizes);
    setActualSizes(gradIn, inActualSizes);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    valType* pFmInput       = (valType*) m_hostBuffers[inTensor];
    valType* pGradIn        = (valType*) m_hostBuffers[gradIn];
    valType* weightsData    = (valType*) m_hostBuffers[weightTensor];
    float*   pRunningMeanIn = (float*) m_hostBuffers[runningMeanIn];
    float*   pRunningIstdIn = (float*) m_hostBuffers[runningIstdIn];
    float*   pGammaIn       = (float*) m_hostBuffers[gammaIn];

    std::vector<float> outConv(sizeInElements);

    float* pGradBetaOut  = (float*) m_hostBuffers[gradBeta];
    float* pGradGammaOut = (float*) m_hostBuffers[gradGamma];
    float* pOutConv      = (float*) m_hostBuffers[outTensor];

    std::vector<valType> gradOutputBufferRef(sizeInElements);
    std::fill(gradOutputBufferRef.begin(), gradOutputBufferRef.end(), 0.0f);
    std::array<float, C> gradBetaRef {};
    std::array<float, C> gradGammaRef {};
    gradBetaRef.fill(0.f);
    gradGammaRef.fill(0.f);

    calcBatchNormBackwardRef<valType>(gradBetaRef.data(),
                                      gradGammaRef.data(),
                                      gradOutputBufferRef.data(),
                                      pFmInput,
                                      pGradIn,
                                      pRunningMeanIn,
                                      pRunningIstdIn,
                                      pGammaIn,
                                      inActualSizes);

    auto inDesc      = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    auto weightDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(weightTensor));
    auto convOutDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(outTensor));

    memcpy(inDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);
    memcpy(convOutDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);

    calculateFwdConvolution(inDesc,
                            (char*)gradOutputBufferRef.data(),
                            weightDesc,
                            (char*)weightsData,
                            convOutDesc,
                            (char*)outConv.data(),
                            params,
                            m_deviceType);

    validateResult(outConv.data(), pOutConv, sizeInElements);
    validateResult(gradBetaRef.data(), pGradBetaOut, C);
    validateResult(gradGammaRef.data(), pGradGammaOut, C);
}

template<typename valType>
void SynGaudiTypedDynamicBnTest::convConsumerBwdTest(size_t height, size_t width, size_t batch)
{
    size_t inActualW      = width;
    size_t inActualH      = height;
    size_t inActualB      = batch;
    size_t sizeInElements = C * inActualW * inActualH * inActualB;

    synDataType synType     = dataTypeToSynType<valType>();
    std::string bn_bwd_guid = getBwdGuid<valType>();
    ASSERT_TRUE(bn_bwd_guid.size() > 0) << "invalid data type";

    unsigned inMaxSizes[]    = {C, MAX_W, MAX_H, MAX_B};
    unsigned inMinSizes[]    = {C, MIN_W, MIN_H, MIN_B};
    unsigned bnTensorSizes[] = {C};
    unsigned weightSizes[]   = {C, C, 1, 1};

    ns_BatchNormKernel::Params bnParams {};
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon     = 1e-05;

    synConvolutionParams params;

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            TENSOR_DIMS,
                                            synType,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned weightTensor =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, weightSizes, TENSOR_DIMS, synType);

    unsigned gradIn = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          inMaxSizes,
                                          TENSOR_DIMS,
                                          synType,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          inMinSizes);

    unsigned convOutTensor = createTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          inMaxSizes,
                                          TENSOR_DIMS,
                                          synType,
                                          nullptr,
                                          inMinSizes);

    unsigned runningMeanIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "running_mean");
    unsigned runningIstdIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_ALL_ONES,
                                                 nullptr,
                                                 bnTensorSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "running_var");
    unsigned gammaIn       = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_ALL_ONES,
                                           nullptr,
                                           bnTensorSizes,
                                           1,
                                           syn_type_float,
                                           nullptr,
                                           "gamma_in");

    unsigned gradBeta  = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            bnTensorSizes,
                                            1,
                                            syn_type_float,
                                            nullptr,
                                            "grad_beta");
    unsigned gradGamma = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             bnTensorSizes,
                                             1,
                                             syn_type_float,
                                             nullptr,
                                             "grad_gamma");

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             inMaxSizes,
                                             TENSOR_DIMS,
                                             synType,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inMinSizes);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {inTensor, weightTensor},
                   {convOutTensor},
                   &params,
                   sizeof(params));

    addNodeToGraph(bn_bwd_guid.c_str(),
                   {convOutTensor, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {outTensor, gradBeta, gradGamma},
                   &bnParams,
                   sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    unsigned inActualSizes[] = {C, inActualW, inActualH, inActualB};
    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, inActualSizes);
    setActualSizes(gradIn, inActualSizes);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float*             pConvRefferenceIn     = nullptr;
    float*             pConvRefferenceweight = nullptr;
    float*             pRefferenceGradIn     = nullptr;
    std::vector<float> convRefferenceIn;
    std::vector<float> convRefferenceWeight;
    std::vector<float> refferenceGradIn;
    std::vector<float> outputConv(sizeInElements);
    if (synType == syn_type_float)
    {
        pConvRefferenceIn     = castHostInBuffer<float>(inTensor);
        pConvRefferenceweight = castHostInBuffer<float>(weightTensor);
        pRefferenceGradIn     = castHostInBuffer<float>(gradIn);
    }
    else if (synType == syn_type_bf16)
    {
        convRefferenceIn.reserve(sizeInElements);
        convRefferenceWeight.reserve(C * C);
        refferenceGradIn.reserve(sizeInElements);

        bfloat16* convInData = castHostInBuffer<bfloat16>(inTensor);
        bfloat16* weightData = castHostInBuffer<bfloat16>(weightTensor);
        bfloat16* gradInData = castHostInBuffer<bfloat16>(gradIn);

        for (unsigned i = 0; i < sizeInElements; i++)
        {
            convRefferenceIn[i] = (float)convInData[i];
            refferenceGradIn[i] = (float)gradInData[i];
        }
        for (unsigned i = 0; i < C * C; i++)
        {
            convRefferenceWeight[i] = (float)weightData[i];
        }
        pConvRefferenceIn     = convRefferenceIn.data();
        pConvRefferenceweight = convRefferenceWeight.data();
        pRefferenceGradIn     = refferenceGradIn.data();
    }
    else
    {
        ASSERT_ANY_THROW() << "invalid data type";
    }

    auto inDesc      = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    auto weightDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(weightTensor));
    auto convOutDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(outTensor));

    memcpy(inDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);
    memcpy(convOutDesc.m_sizes, inActualSizes, sizeof(unsigned) * TENSOR_DIMS);

    inDesc.m_dataType      = syn_type_float;
    weightDesc.m_dataType  = syn_type_float;
    convOutDesc.m_dataType = syn_type_float;

    calculateFwdConvolution(inDesc,
                            (char*)pConvRefferenceIn,
                            weightDesc,
                            (char*)pConvRefferenceweight,
                            convOutDesc,
                            (char*)outputConv.data(),
                            params,
                            m_deviceType);

    float* pRunningMeanIn = (float*) m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*) m_hostBuffers[runningIstdIn];
    float* pGammaIn       = (float*) m_hostBuffers[gammaIn];

    float* pGradBetaOut  = (float*) m_hostBuffers[gradBeta];
    float* pGradGammaOut = (float*) m_hostBuffers[gradGamma];

    std::vector<float> gradOutputBufferRef(sizeInElements);
    std::fill(gradOutputBufferRef.begin(), gradOutputBufferRef.end(), 0.0f);
    std::array<float, C> gradBetaRef {};
    std::array<float, C> gradGammaRef {};
    gradBetaRef.fill(0.f);
    gradGammaRef.fill(0.f);

    valType* bnOut = castHostInBuffer<valType>(outTensor);

    calcBatchNormBackwardRef<float>(gradBetaRef.data(),
                                    gradGammaRef.data(),
                                    gradOutputBufferRef.data(),
                                    outputConv.data(),
                                    pRefferenceGradIn,
                                    pRunningMeanIn,
                                    pRunningIstdIn,
                                    pGammaIn,
                                    inActualSizes);

    validateResult(bnOut, gradOutputBufferRef.data(), sizeInElements);
    validateResult(gradBetaRef.data(), pGradBetaOut, C);
    validateResult(gradGammaRef.data(), pGradGammaOut, C);
}

TEST_P_GC(SynGaudiTypedDynamicBnTest, batch_norm_dynamic_hw_fwd_test_float, {synDeviceGaudi, synDeviceGaudi2})
{
    fwdTest<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest, batch_norm_dynamic_hw_fwd_test_bf16, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    fwdTest<bfloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_producer_batch_norm_dynamic_hw_fwd_test_float,
          {synDeviceGaudi, synDeviceGaudi2})
{
    convProducerFwdTest<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_producer_batch_norm_dynamic_hw_fwd_test_bf16,
          {synDeviceGaudi, synDeviceGaudi2})
{
    convProducerFwdTest<bfloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_consumer_batch_norm_dynamic_hw_fwd_test_float,
          {synDeviceGaudi, synDeviceGaudi2})
{
    convConsumerFwdTest<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_consumer_batch_norm_dynamic_hw_fwd_test_bf16,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    convConsumerFwdTest<bfloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest, batch_norm_dynamic_hw_bwd_test_bf16, {synDeviceGaudi, synDeviceGaudi2,  synDeviceGaudi3})
{
    bwdTest<bfloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest, batch_norm_dynamic_hw_bwd_test_float, {synDeviceGaudi, synDeviceGaudi2})
{
    bwdTest<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_producer_batch_norm_dynamic_hw_bwd_test_float,
          {synDeviceGaudi, synDeviceGaudi2})
{
    convProducerBwdTest<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_producer_batch_norm_dynamic_hw_bwd_test_bf16,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    convProducerBwdTest<bfloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_consumer_batch_norm_dynamic_hw_bwd_test_float,
          {synDeviceGaudi, synDeviceGaudi2})
{
    convConsumerBwdTest<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

TEST_P_GC(SynGaudiTypedDynamicBnTest,
          conv_consumer_batch_norm_dynamic_hw_bwd_test_bf16,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    convConsumerBwdTest<bfloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    DSD,
    SynGaudiTypedDynamicBnTest,
    ::testing::Combine(::testing::Range(SynGaudiTypedDynamicBnTest::MIN_W,
                                        SynGaudiTypedDynamicBnTest::MAX_W + 1,
                                        (SynGaudiTypedDynamicBnTest::MAX_W - SynGaudiTypedDynamicBnTest::MIN_W) / 2),
                       ::testing::Range(SynGaudiTypedDynamicBnTest::MIN_H,
                                        SynGaudiTypedDynamicBnTest::MAX_H + 1,
                                        (SynGaudiTypedDynamicBnTest::MAX_H - SynGaudiTypedDynamicBnTest::MIN_H) / 2),
                       ::testing::Range(SynGaudiTypedDynamicBnTest::MIN_B,
                                        SynGaudiTypedDynamicBnTest::MAX_B + 1,
                                        (SynGaudiTypedDynamicBnTest::MAX_B - SynGaudiTypedDynamicBnTest::MIN_B) / 2)));

TEST_P_GC(SynGaudiDynamicChannelsBatchNormTest, batch_norm_fwd_dynamic_over_channels, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{

    unsigned inMaxSizes[] = {MAX_C, WIDTH, HEIGHT, BATCH};
    unsigned inMinSizes[] = {MIN_C, WIDTH, HEIGHT, BATCH};
    const unsigned ACTUAL_C = GetParam();

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, inMinSizes);

    unsigned bnChannelMinSizes[] = {MIN_C};
    unsigned bnChannelMaxSizes[] = {MAX_C};

    unsigned bnBetaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                bnChannelMaxSizes, 1, syn_type_float, nullptr, "Beta",
                                                0, 0, nullptr, bnChannelMinSizes);
    unsigned bnGammaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                 bnChannelMaxSizes, 1, syn_type_float, nullptr, "Gamma",
                                                 0, 0, nullptr, bnChannelMinSizes);
    unsigned bnInMeanTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                  bnChannelMaxSizes, 1, syn_type_float, nullptr, "InMean",
                                                  0, 0, nullptr, bnChannelMinSizes);
    unsigned bnInVarTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                 bnChannelMaxSizes, 1, syn_type_float, nullptr, "InVar",
                                                 0, 0, nullptr, bnChannelMinSizes);

    unsigned bnOutMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                   bnChannelMaxSizes, 1, syn_type_float, nullptr,
                                                    bnChannelMinSizes);
    unsigned bnOutStdTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                           bnChannelMaxSizes, 1, syn_type_float, nullptr,
                                           bnChannelMinSizes);
    unsigned bnOutRunMeanTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               bnChannelMaxSizes, 1, syn_type_float, nullptr,
                                               bnChannelMinSizes);
    unsigned bnOutRunVarTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                              bnChannelMaxSizes, 1, syn_type_float, nullptr,
                                              bnChannelMinSizes);
    ns_BatchNormKernel::Params bnParams {};
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, inMinSizes);


    addNodeToGraph("batch_norm_fwd_f32",
                   {inTensor, bnBetaTensor, bnGammaTensor, bnInMeanTensor, bnInVarTensor},
                   {outTensor, bnOutMeanTensor, bnOutStdTensor, bnOutRunMeanTensor, bnOutRunVarTensor},
                   &bnParams, sizeof bnParams);

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    unsigned inActualSizes[] = {ACTUAL_C, WIDTH, HEIGHT, BATCH};
    unsigned cActualSizes[] = {ACTUAL_C};
    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, inActualSizes);
    setActualSizes(bnBetaTensor, cActualSizes);
    setActualSizes(bnGammaTensor, cActualSizes);
    setActualSizes(bnInMeanTensor, cActualSizes);
    setActualSizes(bnInVarTensor, cActualSizes);
    setActualSizes(bnOutMeanTensor, cActualSizes);
    setActualSizes(bnOutStdTensor, cActualSizes);
    setActualSizes(bnOutRunMeanTensor, cActualSizes);
    setActualSizes(bnOutRunVarTensor, cActualSizes);

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    auto totalMaxSize = multiplyElements(inMaxSizes, inMaxSizes + TENSOR_DIMS);
    std::vector<float> mean;
    mean.resize(ACTUAL_C);
    std::vector<float> runningMean;
    runningMean.resize(ACTUAL_C);
    std::fill(mean.begin(), mean.end(), 0.0f);
    std::fill(runningMean.begin(), runningMean.end(), 0.0f);

    std::vector<float> iStd, runningVar;
    iStd.resize(ACTUAL_C);
    runningVar.resize(ACTUAL_C);
    std::fill(iStd.begin(), iStd.end(), 1.0f);
    std::fill(runningVar.begin(), runningVar.end(), 1.0f);



    std::vector<float> outputRef(totalMaxSize);
    std::fill(outputRef.begin(), outputRef.end(), 0.0f);

    calcBatchNormForwardRef<float>(mean.data(),
                                   iStd.data(),
                                   runningMean.data(),
                                   runningVar.data(),
                                   outputRef.data(),
                                   bnParams.momentum,
                                   inData,
                                   castHostInBuffer<float>(bnBetaTensor),
                                   castHostInBuffer<float>(bnGammaTensor),
                                   inActualSizes);

    auto* output = castHostOutBuffer<float>(outTensor);

    constexpr float eps = .01f;
    for (int i = 0; i < totalMaxSize; i++)
    {
        ASSERT_TRUE(float_eq(output[i], outputRef[i], eps)) << "Accuracy error (Eps = " << eps << ")"
                                                            << " index = " << i
                                                            << " output = " << output[i]
                                                            << " reference = " << outputRef[i];
    }
}

TEST_P_GC(SynGaudiDynamicChannelsBatchNormTest, batch_norm_bwd_dynamic_over_channels, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned ACTUAL_C = GetParam();
    size_t sizeInElements = ACTUAL_C * WIDTH * HEIGHT * BATCH;

    unsigned inMaxSizes[] = {MAX_C, WIDTH, HEIGHT, BATCH};
    unsigned inMinSizes[] = {MIN_C, WIDTH, HEIGHT, BATCH};
    unsigned bnMaxTensorSizes[] = {MAX_C};
    unsigned bnMinTensorSizes[] = {MIN_C};
    ns_BatchNormKernel::Params bnParams{};
    bnParams.momentum = 0.1;
    bnParams.threshold.f = 1e-05;
    bnParams.epsilon = 1e-05;

    unsigned featureMapIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                                inMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, nullptr,
                                                0, 0, nullptr, inMinSizes);
    unsigned gradIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                          inMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, nullptr,
                                          0, 0, nullptr, inMinSizes);

    unsigned runningMeanIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                  bnMaxTensorSizes, 1, syn_type_float, nullptr,
                                                  "running_mean", 0, 0, nullptr, bnMinTensorSizes);
    unsigned runningIstdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr,
                                                  bnMaxTensorSizes ,1, syn_type_float, nullptr,
                                                  "running_var", 0, 0, nullptr, bnMinTensorSizes);
    unsigned gammaIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr,
                                                  bnMaxTensorSizes ,1, syn_type_float, nullptr,
                                                  "gamma_in", 0, 0, nullptr, bnMinTensorSizes);



    unsigned gradBeta     = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                bnMaxTensorSizes, 1,  syn_type_float, nullptr,
                                                "grad_beta", 0, 0, nullptr, bnMinTensorSizes);
    unsigned gradGamma    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                bnMaxTensorSizes, 1, syn_type_float, nullptr,
                                                "grad_gamma", 0, 0, nullptr, bnMinTensorSizes);

    unsigned gradOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                           inMaxSizes, TENSOR_DIMS, syn_type_float, nullptr, nullptr,
                                           0, 0, nullptr, inMinSizes);


    addNodeToGraph("batch_norm_bwd_f32", {featureMapIn, gradIn, runningMeanIn, runningIstdIn, gammaIn},
                   {gradOut, gradBeta, gradGamma}, &bnParams, sizeof(ns_BatchNormKernel::Params));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    unsigned channelActualSizes[] = {ACTUAL_C};

    unsigned inActualSizes[] = {ACTUAL_C, WIDTH, HEIGHT, BATCH};
    setActualSizes(featureMapIn, inActualSizes);
    setActualSizes(gradOut, inActualSizes);
    setActualSizes(gradIn, inActualSizes);
    setActualSizes(runningMeanIn, channelActualSizes);
    setActualSizes(runningIstdIn, channelActualSizes);
    setActualSizes(gammaIn, channelActualSizes);
    setActualSizes(gradBeta, channelActualSizes);
    setActualSizes(gradGamma, channelActualSizes);
    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";


    float* pFmInput = (float*)m_hostBuffers[featureMapIn];
    float* pGradIn = (float*)m_hostBuffers[gradIn];
    float* pRunningMeanIn = (float*)m_hostBuffers[runningMeanIn];
    float* pRunningIstdIn = (float*)m_hostBuffers[runningIstdIn];
    float* pGammaIn = (float*)m_hostBuffers[gammaIn];

    float* pGradOut = (float*)m_hostBuffers[gradOut];
    float* pGradBetaOut = (float*)m_hostBuffers[gradBeta];
    float* pGradGammaOut = (float*)m_hostBuffers[gradGamma];


    std::vector<float> gradBetaRef, gradGammaRef;
    gradBetaRef.resize(ACTUAL_C);
    gradGammaRef.resize(ACTUAL_C);
    std::fill(gradBetaRef.begin(), gradBetaRef.end(), 0.0f);
    std::fill(gradGammaRef.begin(), gradGammaRef.end(), 0.0f);
    std::vector<float> gradOutputBufferRef(sizeInElements);

    calcBatchNormBackwardRef<float>(gradBetaRef.data(),
                                    gradGammaRef.data(),
                                    gradOutputBufferRef.data(),
                                    pFmInput,
                                    pGradIn,
                                    pRunningMeanIn,
                                    pRunningIstdIn,
                                    pGammaIn,
                                    inActualSizes);

    validateResult(gradOutputBufferRef.data(), pGradOut, sizeInElements);
    validateResult(gradBetaRef.data(), pGradBetaOut, ACTUAL_C);
    validateResult(gradGammaRef.data(), pGradGammaOut, ACTUAL_C);



}

INSTANTIATE_TEST_SUITE_P(DSD,
                         SynGaudiDynamicChannelsBatchNormTest,
                         ::testing::Range<unsigned>(SynGaudiDynamicChannelsBatchNormTest::MIN_C,
                                                    SynGaudiDynamicChannelsBatchNormTest::MAX_C,
                                                    (SynGaudiDynamicChannelsBatchNormTest::MIN_C +
                                                     SynGaudiDynamicChannelsBatchNormTest::MAX_C) /
                                                        4));
