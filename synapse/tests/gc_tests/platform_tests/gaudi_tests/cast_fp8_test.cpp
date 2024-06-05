#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "quantization_data.h"
#include "synapse_common_types.h"
#include "gtest/gtest.h"
#include "synapse_api_types.h"
#include "utils.h"
#include "syn_singleton.hpp"
#include <data_types/bfloat16.h>
#include <data_types/fp8.h>

struct Fp8TestData
{
    synDataType        outDataType;
    std::vector<float> expected;
    unsigned int       expBias;
    std::vector<float> castInput;
};

class SynTrainingTestInfraFp8Cast : public SynTrainingTestInfra
{
protected:
    virtual void SetUpTest() override
    {
        SynTrainingTestInfra::SetUpTest();
        pushGlobalConf("ENABLE_CONSTANT_FOLDING", "true");
        pushGlobalConf("ENABLE_CALC_DYNAMIC_RANGE", "true");
    }
    template<typename CppType>
    void checkExpectedResult(unsigned expBias, std::vector<float> expected, size_t tensorOutIndex)
    {
        CppType* outputBuffer = castHostOutBuffer<CppType>(tensorOutIndex);
        for (uint64_t i = 0; i < getTensorElementCount(tensorOutIndex); ++i)
        {
            ASSERT_EQ(expected[i], outputBuffer[i].toFloat(expBias));
        }
    }

    float getMaxAbsValue(std::vector<float> input)
    {
        float maxVal = abs(input[0]);
        for (float val : input)
        {
            if (abs(val) > maxVal)
            {
                maxVal = abs(val);
            }
        }
        return maxVal;
    }
};

class SynTrainingTestInfraFp8Tests
: public SynTrainingTestInfraFp8Cast
, public testing::WithParamInterface<Fp8TestData>
{
};

TEST_P_GC(SynTrainingTestInfraFp8Tests, add_and_cast, {synDeviceGaudi2})
{
    const unsigned           dims        = 2;
    unsigned                 sizes[]     = {7, 1};
    const synDataType        inDataType  = syn_type_float;
    const synDataType        outDataType = GetParam().outDataType;
    const std::vector<float> expected    = GetParam().expected;
    unsigned int             expBias     = GetParam().expBias;
    std::vector<float>       inputBuffer1 {-65, -10, -7, -5, 2, 0.5, 4};
    std::vector<float>       inputBuffer2 {-1, -9, -0.5, 0, -2, 6, 5};
    // output of add will be {-66, -18, -7.5, -5, 0, 6.5, 9} in float

    // create graph
    setGraphInferenceModeAndQuantizationEnabled();
    unsigned T1 = createConstPersistTensor(TensorUsage::INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           inputBuffer1.data(),
                                           sizes,
                                           dims,
                                           inDataType,
                                           nullptr,
                                           "addInTensor1");

    unsigned T2 = createConstPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           inputBuffer2.data(),
                                           sizes,
                                           dims,
                                           inDataType,
                                           nullptr,
                                           "addInTensor2");
    unsigned T3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, inDataType);

    auto T4 = createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims, outDataType);

    // set cast's dynamic range
    synQuantDynamicRange dynamicRange;
    dynamicRange.max = 9;
    dynamicRange.min = -66;

    setTensorQuantizationData(T4, SYN_QUANT_DYNAMIC_RANGE, &dynamicRange, sizeof(synQuantDynamicRange));

    addNodeToGraph("add_fwd_f32", {T1, T2}, {T3});
    addNodeToGraph(getCastGUID(inDataType, outDataType).c_str(), {T3}, {T4});

    compileTopology();
    runTopology();

    if (outDataType == syn_type_fp8_152)
    {
        checkExpectedResult<fp8_152_t>(expBias, expected, T4);
    }
    else
    {
        checkExpectedResult<fp8_143_t>(expBias, expected, T4);
    }
}
//
INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingTestInfraFp8Tests,
                         ::testing::Values(Fp8TestData {syn_type_fp8_152, {-64, -20, -8, -5, 0, 6, 8}, 15},
                                           Fp8TestData {syn_type_fp8_143, {-64, -20, -7.5, -5, 0, 6.5, 9}, 7}));

// TODO SW-173110 decide if to remove those tests or refactor for new flow
class DISABLED_SynTrainingTestInfraFp8BiasTests
: public SynTrainingTestInfraFp8Cast
, public testing::WithParamInterface<Fp8TestData>
{
};

TEST_P_GC(DISABLED_SynTrainingTestInfraFp8BiasTests, cast_fp8_differnt_exp_bias_data_range, {synDeviceGaudi2})
{

    const unsigned           dims        = 1;
    const synDataType        inDataType  = syn_type_float;
    const synDataType        outDataType = GetParam().outDataType;
    std::vector<float>       inputBuffer = GetParam().castInput;
    const std::vector<float> expected    = GetParam().expected;
    unsigned int             expBias     = GetParam().expBias;
    unsigned int             sizes       = inputBuffer.size();

    // create graph
    setGraphInferenceModeAndQuantizationEnabled();
    unsigned T1 = createConstPersistTensor(TensorUsage::INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           inputBuffer.data(),
                                           &sizes,
                                           dims,
                                           inDataType,
                                           nullptr,
                                           "CastInTensor");
    auto T2 = createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, &sizes, dims, outDataType);

    double               maxVal = getMaxAbsValue(inputBuffer);
    synQuantDynamicRange dynamicRange;
    dynamicRange.max = maxVal;

    setTensorQuantizationData(T2, SYN_QUANT_DYNAMIC_RANGE, &dynamicRange, sizeof(synQuantDynamicRange));

    addNodeToGraph(getCastGUID(inDataType, outDataType).c_str(), {T1}, {T2});

    compileTopology();
    runTopology();

    if (outDataType == syn_type_fp8_143)
    {
        checkExpectedResult<fp8_143_t>(expBias, expected, T2);
    }
}

TEST_P_GC(DISABLED_SynTrainingTestInfraFp8BiasTests, cast_fp8_differnt_exp_bias_quant_metadata, {synDeviceGaudi2})
{
    const unsigned           dims        = 1;
    const synDataType        inDataType  = syn_type_float;
    const synDataType        outDataType = GetParam().outDataType;
    std::vector<float>       inputBuffer = GetParam().castInput;
    const std::vector<float> expected    = GetParam().expected;
    unsigned int             expBias     = GetParam().expBias;
    unsigned int             sizes       = inputBuffer.size();

    // create graph
    setGraphInferenceModeAndQuantizationEnabled();
    unsigned T1 = createConstPersistTensor(TensorUsage::INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           inputBuffer.data(),
                                           &sizes,
                                           dims,
                                           inDataType,
                                           nullptr,
                                           "CastInTensor");
    auto T2 = createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, &sizes, dims, outDataType);

    synFpQuantMetadata metaData;
    metaData.dataType         = outDataType;
    metaData.numFpQuantParams = 1;
    synFpQuantParam quantParam;
    quantParam.expBias     = expBias;
    quantParam.scale       = 1;
    metaData.fpQuantParams = &quantParam;

    setTensorQuantizationData(T2, SYN_FP_QUANT_METADATA, &metaData, sizeof(synFpQuantMetadata));

    addNodeToGraph(getCastGUID(inDataType, outDataType).c_str(), {T1}, {T2});

    compileTopology();
    runTopology();

    if (outDataType == syn_type_fp8_143)
    {
        checkExpectedResult<fp8_143_t>(expBias, expected, T2);
    }
}

TEST_P_GC(DISABLED_SynTrainingTestInfraFp8BiasTests, cast_fp8_differnt_exp_bias_folding, {synDeviceGaudi2})
{
    const unsigned           dims        = 2;
    const synDataType        inDataType  = syn_type_float;
    const synDataType        outDataType = GetParam().outDataType;
    std::vector<float>       inputBuffer = GetParam().castInput;
    const std::vector<float> expected    = GetParam().expected;
    unsigned int             expBias     = GetParam().expBias;
    std::vector<unsigned>    inSizes     = {inputBuffer.size(), 1};
    std::vector<unsigned>    outSizes    = {1, inputBuffer.size()};

    // create graph
    setGraphInferenceModeAndQuantizationEnabled();
    unsigned T1 = createConstPersistTensor(TensorUsage::INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           inputBuffer.data(),
                                           inSizes.data(),
                                           dims,
                                           inDataType,
                                           nullptr,
                                           "CastInTensor");
    auto T2 = createTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inSizes.data(), dims, outDataType);
    auto T3 =
        createPersistTensor(TensorUsage::OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), dims, outDataType);

    synFpQuantMetadata metaData;
    metaData.dataType         = outDataType;
    metaData.numFpQuantParams = 1;
    synFpQuantParam quantParam;
    quantParam.expBias     = expBias;
    quantParam.scale       = 1;
    metaData.fpQuantParams = &quantParam;

    setTensorQuantizationData(T2, SYN_FP_QUANT_METADATA, &metaData, sizeof(synFpQuantMetadata));

    addNodeToGraph(getCastGUID(inDataType, outDataType).c_str(), {T1}, {T2});
    addNodeToGraph("reshape", {T2}, {T3});

    compileTopology();
    runTopology();

    if (outDataType == syn_type_fp8_143)
    {
        checkExpectedResult<fp8_143_t>(expBias, expected, T3);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    DISABLED_SynTrainingTestInfraFp8BiasTests,
    ::testing::Values(
        Fp8TestData {syn_type_fp8_143, {-26, -20, -7.5, -5, 0, 6.5, 9, 3840}, 3, {-26, -20, -7.5, -5, 0, 6.5, 9, 3800}},
        Fp8TestData {syn_type_fp8_143,
                     {-14, -11, -7.5, -5, 0, 6.5, 9, 9},
                     11,
                     {-14, -11.23, -7.5, -5, 0, 6.5, 8.76, 9}},
        Fp8TestData {syn_type_fp8_143,
                     {-0.9375, -0.8125, -0.625, -0.40625, -0.1015625, 0, 0.203125, 0.8125},
                     15,
                     {-0.93, -0.8, -0.6, -0.40625, -0.1, 0, 0.2, 0.8}}));

class SynCastScaleInjectionTests
: public SynTrainingTestInfraFp8Cast
, public testing::WithParamInterface<synDataType>

{
public:
    void runTest(synDataType inDataType, bool constFolding, bool pcScaling)
    {
        std::string constantFolding   = constFolding ? "true" : "false";
        std::string perChannelScaling = pcScaling ? "true" : "false";
        pushGlobalConf("ENABLE_CONSTANT_FOLDING", constantFolding);
        pushGlobalConf("PER_CHANNEL_SCALING", perChannelScaling);

        const synDataType        outDataType  = syn_type_bf16;
        std::vector<float>       inputBuffer1 = {7680, 1000, 4000, 0, -7680, 2000};
        std::vector<float>       inputBuffer2 = {0.234375, 0.1171875, 0.234375, 0.1171875, 0.234375, 0.234375};
        const std::vector<float> expected     = {3008, 1984, -1320, -420};
        std::vector<unsigned>    inSizes1     = {3, 2};
        std::vector<unsigned>    inSizes2     = {2, 3};
        std::vector<unsigned>    outSizes     = {2, 2};

        // create graph
        setGraphInferenceModeAndQuantizationEnabled();
        unsigned T1 = createConstPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               inputBuffer1.data(),
                                               inSizes1.data(),
                                               inSizes1.size(),
                                               inDataType,
                                               nullptr,
                                               "Input1");

        unsigned T2 = createConstPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_FROM_INITIALIZER,
                                               inputBuffer2.data(),
                                               inSizes2.data(),
                                               inSizes2.size(),
                                               inDataType,
                                               nullptr,
                                               "Input2");

        unsigned T3 = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          outSizes.data(),
                                          outSizes.size(),
                                          outDataType,
                                          nullptr,
                                          "Output");

        synQuantDynamicRange dynamicRangeInput1;
        double               maxValInput1 = getMaxAbsValue(inputBuffer1);
        dynamicRangeInput1.max            = maxValInput1;
        dynamicRangeInput1.min            = maxValInput1 * -1;
        synQuantDynamicRange dynamicRangeInput2;
        double               maxValInput2 = getMaxAbsValue(inputBuffer2);
        dynamicRangeInput2.max            = maxValInput2;
        dynamicRangeInput2.min            = maxValInput2 * -1;

        setTensorQuantizationData(T1, SYN_QUANT_DYNAMIC_RANGE, &dynamicRangeInput1, sizeof(synQuantDynamicRange));
        setTensorQuantizationData(T2, SYN_QUANT_DYNAMIC_RANGE, &dynamicRangeInput2, sizeof(synQuantDynamicRange));

        addNodeToGraph("gemm", {T1, T2}, {T3});

        compileTopology();
        runTopology();

        const HabanaGraph* graph               = synSingleton::getInstanceInternal()->getGraph(getGraph(0).graphHandle);
        unsigned           scaling_nodes_cnt   = 0;
        unsigned           descaling_nodes_cnt = 0;
        for (const NodePtr& n : graph->getNodes())
        {
            if (n->getGUID().find("mult") != std::string::npos)
            {
                scaling_nodes_cnt++;
            }
            else if (n->getGUID().find("div") != std::string::npos)
            {
                descaling_nodes_cnt++;
            }
        }
        if (constFolding)
        {
            ASSERT_EQ(scaling_nodes_cnt, 0);
        }
        else
        {
            ASSERT_EQ(scaling_nodes_cnt, 2);
        }
        ASSERT_EQ(descaling_nodes_cnt, 1);

        bfloat16* outputBuffer = castHostOutBuffer<bfloat16>(T3);
        for (uint64_t i = 0; i < getTensorElementCount(T3); ++i)
        {
            ASSERT_EQ(expected[i], float(outputBuffer[i]));
        }
    }
};

TEST_P_GC(SynCastScaleInjectionTests, scaling_nodes_for_mme_fp8, {synDeviceGaudi2})
{
    runTest(GetParam(), false, false);
}

TEST_P_GC(SynCastScaleInjectionTests, scaling_PC_nodes_for_mme_fp8, {synDeviceGaudi2})
{
    runTest(GetParam(), false, true);
}

TEST_P_GC(SynCastScaleInjectionTests, fold_scaling_nodes_for_mme_fp8, {synDeviceGaudi2})
{
    runTest(GetParam(), true, false);
}

TEST_P_GC(SynCastScaleInjectionTests, fold_scaling_PC_nodes_for_mme_fp8, {synDeviceGaudi2})
{
    runTest(GetParam(), true, true);
}

INSTANTIATE_TEST_SUITE_P(, SynCastScaleInjectionTests, ::testing::Values(syn_type_float, syn_type_bf16));
