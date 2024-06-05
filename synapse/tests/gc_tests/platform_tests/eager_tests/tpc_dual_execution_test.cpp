#include "defs.h"
#include "fmt-9.1.0/include/fmt/core.h"
#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "synapse_common_types.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <data_types/bfloat16.h>

class SynTrainingTpcDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
    template<typename T>
    T getBufferValue(unsigned bufferIndex, int index, synDataType dataType)
    {
        void* buffer = m_hostBuffers[bufferIndex];
        switch (dataType)
        {
            case syn_type_int8:
                return static_cast<int8_t*>(buffer)[index];
            case syn_type_uint8:
                return static_cast<uint8_t*>(buffer)[index];
            case syn_type_int16:
                return static_cast<int16_t*>(buffer)[index];
            case syn_type_uint16:
                return static_cast<uint16_t*>(buffer)[index];
            case syn_type_int32:
                return static_cast<int32_t*>(buffer)[index];
            case syn_type_uint32:
                return static_cast<uint32_t*>(buffer)[index];
            case syn_type_bf16:
                return Bfloat16(static_cast<uint16_t*>(buffer)[index]).toFloat();
            case syn_type_fp16:
                return HalfFloat<true>(static_cast<uint16_t*>(buffer)[index]).toFloat();
            case syn_type_float:
                return static_cast<float*>(buffer)[index];
            default:
                HB_ASSERT(false, "unsupported type");
                return 0;
        }
    }

public:
    void runExtractedAddWithAlpha(synDataType dataType);
    void runAddWithAlpha(synDataType dataType);
    void runConstant(synDataType dataType, fint_t constant);
    template<typename CastKernelParams = ns_CastKernel::Params>
    void runCast(synDataType fromDataType, synDataType toDataType, fint_t constant, CastF32RoundMode_t roundingMode);
    void runConstantCastAdd(synDataType fromDataType, synDataType toDataType, fint_t constant);
    SynTrainingTpcDualExecutionTest() { ReleaseDevice(); }
};

constexpr auto getIntVal = [](int val) {
    fint_t constantVal = {};
    constantVal.i      = val;
    return constantVal;
};

void SynTrainingTpcDualExecutionTest::runAddWithAlpha(synDataType dataType)
{
    std::array<unsigned, 3> sizes = {128, 32, 24};

    createPersistTensors(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), sizes.size(), dataType);
    createPersistTensors(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes.data(), sizes.size(), dataType);
    auto addOut = createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), dataType);
    ns_BinaryWithAlphaKernel::Params params = {};
    if (dataType != syn_type_uint32 && dataType != syn_type_int32)
    {
        params.alpha.f = 2.0;
    }
    else
    {
        params.alpha.i = 2;
    }
    params.mode = BINARY_WITH_ALPHA_MODE_ADD;

    addNodesToGraphs(fmt::format("binary_with_alpha_fwd_{}", getDtypeSuffixFromSynDataType(dataType)).c_str(),
                     &params,
                     sizeof(params));

    compileAndRun();

    for (uint64_t i = 0; i < getNumberOfElements(sizes.data(), sizes.size()); i++)
    {
        if (isTypeFloat(dataType))
        {
            float eagerVal = getBufferValue<float>(addOut.eager, i, dataType);
            float graphVal = getBufferValue<float>(addOut.graph, i, dataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
        else
        {
            int64_t eagerVal = getBufferValue<int64_t>(addOut.eager, i, dataType);
            int64_t graphVal = getBufferValue<int64_t>(addOut.graph, i, dataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_binary_with_alpha_fwd_bf16)
{
    runAddWithAlpha(synDataType::syn_type_bf16);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_binary_with_alpha_fwd_f32)
{
    runAddWithAlpha(synDataType::syn_type_float);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_binary_with_alpha_fwd_int16)
{
    runAddWithAlpha(synDataType::syn_type_int16);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_binary_with_alpha_fwd_int32)
{
    runAddWithAlpha(synDataType::syn_type_int32);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_binary_with_alpha_fwd_fp16)
{
    runAddWithAlpha(synDataType::syn_type_fp16);
}

void SynTrainingTpcDualExecutionTest::runExtractedAddWithAlpha(synDataType dataType)
{
    std::array<unsigned, 3> sizes      = {128, 32, 24};
    std::array<unsigned, 1> alphaSizes = {1};

    auto operand1 = createPersistTensors(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sizes.data(),
                                         sizes.size(),
                                         dataType);
    auto operand2 = createPersistTensors(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sizes.data(),
                                         sizes.size(),
                                         dataType);
    auto alpha =
        createTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, alphaSizes.data(), alphaSizes.size(), dataType);

    auto multOutput = createTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), dataType);

    auto addOut = createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), dataType);
    ns_ConstantKernel::Params params = {};
    if (dataType != syn_type_uint32 && dataType != syn_type_int32)
    {
        params.constant.f = 2.0;
    }
    else
    {
        params.constant.i = 2;
    }

    TensorIndicesPair constantOutIndices = {{alpha.graph}, {alpha.eager}};
    addNodesToGraphs(fmt::format("constant_{}", getDtypeSuffixFromSynDataType(dataType)).c_str(),
                     {},
                     constantOutIndices,
                     &params,
                     sizeof(params));

    TensorIndicesPair multInIndices  = {{alpha.graph, operand2.graph}, {alpha.eager, operand2.eager}};
    TensorIndicesPair multOutIndices = {{multOutput.graph}, {multOutput.eager}};
    addNodesToGraphs(fmt::format("mult_fwd_{}", getDtypeSuffixFromSynDataType(dataType)).c_str(),
                     multInIndices,
                     multOutIndices);

    TensorIndicesPair addInIndices  = {{operand1.graph, multOutput.graph}, {operand1.eager, multOutput.eager}};
    TensorIndicesPair addOutIndices = {{addOut.graph}, {addOut.eager}};
    addNodesToGraphs(fmt::format("add_fwd_{}", getDtypeSuffixFromSynDataType(dataType)).c_str(),
                     addInIndices,
                     addOutIndices);

    compileAndRun();

    for (uint64_t i = 0; i < getNumberOfElements(sizes.data(), sizes.size()); i++)
    {
        if (isTypeFloat(dataType))
        {
            float eagerVal = getBufferValue<float>(addOut.eager, i, dataType);
            float graphVal = getBufferValue<float>(addOut.graph, i, dataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
        else
        {
            int64_t eagerVal = getBufferValue<int64_t>(addOut.eager, i, dataType);
            int64_t graphVal = getBufferValue<int64_t>(addOut.graph, i, dataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_int8)
{
    runExtractedAddWithAlpha(synDataType::syn_type_int8);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_uint8)
{
    runExtractedAddWithAlpha(synDataType::syn_type_uint8);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_int16)
{
    runExtractedAddWithAlpha(synDataType::syn_type_int16);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_uint16)
{
    runExtractedAddWithAlpha(synDataType::syn_type_int16);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_int32)
{
    runExtractedAddWithAlpha(synDataType::syn_type_int32);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_bf16)
{
    runExtractedAddWithAlpha(synDataType::syn_type_bf16);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_fp16)
{
    runExtractedAddWithAlpha(synDataType::syn_type_fp16);
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_add_with_alpha_fwd_float)
{
    runExtractedAddWithAlpha(synDataType::syn_type_float);
}

void SynTrainingTpcDualExecutionTest::runConstant(synDataType dataType, fint_t constant)
{
    std::array<unsigned, 1> constantSizes = {4};

    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    GraphIndexPair graphIndexPair = createNewGraphPair();

    auto constantOut = createTensors(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     constantSizes.data(),
                                     constantSizes.size(),
                                     dataType,
                                     nullptr,
                                     graphIndexPair);

    auto memcpyOut = createPersistTensors(OUTPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          constantSizes.data(),
                                          constantSizes.size(),
                                          dataType,
                                          false,
                                          nullptr,
                                          nullptr,
                                          graphIndexPair);

    ns_ConstantKernel::Params params             = {constant};
    TensorIndicesPair         constantOutIndices = {{constantOut.graph}, {constantOut.eager}};
    addNodesToGraphs(fmt::format("constant_{}", getDtypeSuffixFromSynDataType(dataType)).c_str(),
                     {},
                     constantOutIndices,
                     &params,
                     sizeof(params),
                     nullptr,
                     graphIndexPair);

    TensorIndicesPair memcpyInIndices  = {{constantOut.graph}, {constantOut.eager}};
    TensorIndicesPair memcpyOutIndices = {{memcpyOut.graph}, {memcpyOut.eager}};
    addNodesToGraphs(fmt::format("memcpy_{}", getDtypeSuffixFromSynDataType(dataType)).c_str(),
                     memcpyInIndices,
                     memcpyOutIndices,
                     nullptr,
                     0,
                     nullptr,
                     graphIndexPair);

    compileTopology("topology", graphIndexPair);
    runTopology(graphIndexPair);

    for (uint64_t i = 0; i < getNumberOfElements(constantSizes.data(), constantSizes.size()); i++)
    {
        if (isTypeFloat(dataType))
        {
            float eagerVal = getBufferValue<float>(memcpyOut.eager, i, dataType);
            float graphVal = getBufferValue<float>(memcpyOut.graph, i, dataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
        else
        {
            int64_t eagerVal = getBufferValue<int64_t>(memcpyOut.eager, i, dataType);
            int64_t graphVal = getBufferValue<int64_t>(memcpyOut.graph, i, dataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_coverage_int)
{
    for (synDataType dataType : {syn_type_int8, syn_type_uint8, syn_type_int16, syn_type_uint16})
    {
        runConstant(dataType, {65535});
        runConstant(dataType, {65536});
        runConstant(dataType, {2.4});
        runConstant(dataType, {2.5});
        runConstant(dataType, {-2.6});
        runConstant(dataType, {-1});
        runConstant(dataType, {-256});
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_coverage_float)
{
    for (synDataType dataType : {synDataType::syn_type_float, synDataType::syn_type_bf16, synDataType::syn_type_fp16})
    {
        runConstant(dataType, {65536});
        runConstant(dataType, {2.5});
        runConstant(dataType, {8.0 / 3});
        runConstant(dataType, {-11.0 / 3});
        runConstant(dataType, {3.5555555555555555});
        runConstant(dataType, {-1});
        runConstant(dataType, {-0.111111111111});
        runConstant(dataType, {std::numeric_limits<float>::min()});
        runConstant(dataType, {std::numeric_limits<float>::max()});
        runConstant(dataType, {std::numeric_limits<float>::infinity()});
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_coverage_int32)
{
    for (synDataType dataType : {syn_type_int32, syn_type_uint32})
    {
        runConstant(dataType, getIntVal(5));
        runConstant(dataType, getIntVal(-1));
    }
}

template<typename CastKernelParams>
void SynTrainingTpcDualExecutionTest::runCast(synDataType        fromDataType,
                                              synDataType        toDataType,
                                              fint_t             constant,
                                              CastF32RoundMode_t roundingMode)
{
    std::array<unsigned, 1> constantSizes = {1};

    // we can't use the default graphs since this helper function is supposed
    // to be called several times per test.
    GraphIndexPair graphIndexPair = createNewGraphPair();

    TensorIndexPair constantOut;
    if (fromDataType == synDataType::syn_type_int32 || fromDataType == synDataType::syn_type_uint32)
    {
        constantOut = createConstTensors(MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                         &constant.f,
                                         constantSizes.data(),
                                         constantSizes.size(),
                                         fromDataType,
                                         nullptr,
                                         graphIndexPair);
    }
    else
    {
        constantOut = createConstTensors(MEM_INIT_FROM_INITIALIZER,
                                         &constant.f,
                                         constantSizes.data(),
                                         constantSizes.size(),
                                         fromDataType,
                                         nullptr,
                                         graphIndexPair);
    }

    auto castOut = createTensors(OUTPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 constantSizes.data(),
                                 constantSizes.size(),
                                 toDataType,
                                 nullptr,
                                 graphIndexPair);

    auto memcpyOut = createPersistTensors(OUTPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          constantSizes.data(),
                                          constantSizes.size(),
                                          toDataType,
                                          false,
                                          nullptr,
                                          nullptr,
                                          graphIndexPair);

    TensorIndicesPair     castInIndices  = {{constantOut.graph}, {constantOut.eager}};
    TensorIndicesPair     castOutIndices = {{castOut.graph}, {castOut.eager}};
    CastKernelParams      castParams     = {};
    castParams.round_mode                = roundingMode;
    addNodesToGraphs(fmt::format("cast_{}_to_{}",
                                 getDtypeSuffixFromSynDataType(fromDataType),
                                 getDtypeSuffixFromSynDataType(toDataType))
                         .c_str(),
                     castInIndices,
                     castOutIndices,
                     &castParams,
                     sizeof(castParams),
                     nullptr,
                     graphIndexPair);

    TensorIndicesPair memcpyInIndices  = {{castOut.graph}, {castOut.eager}};
    TensorIndicesPair memcpyOutIndices = {{memcpyOut.graph}, {memcpyOut.eager}};
    addNodesToGraphs(fmt::format("memcpy_{}", getDtypeSuffixFromSynDataType(toDataType)).c_str(),
                     memcpyInIndices,
                     memcpyOutIndices,
                     nullptr,
                     0,
                     nullptr,
                     graphIndexPair);

    compileTopology("topology", graphIndexPair);
    runTopology(graphIndexPair);

    for (uint64_t i = 0; i < getNumberOfElements(constantSizes.data(), constantSizes.size()); i++)
    {
        if (isTypeFloat(toDataType))
        {
            float eagerVal = getBufferValue<float>(memcpyOut.eager, i, toDataType);
            float graphVal = getBufferValue<float>(memcpyOut.graph, i, toDataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
        else
        {
            int64_t eagerVal = getBufferValue<int64_t>(memcpyOut.eager, i, toDataType);
            int64_t graphVal = getBufferValue<int64_t>(memcpyOut.graph, i, toDataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_float_to_all)
{
    runCast<ns_CastKernel::ParamsV4>(synDataType::syn_type_float,
                                     synDataType::syn_type_int32,
                                     {std::numeric_limits<float>::max()},
                                     CAST_ROUND_HALF_NE);

    for (CastF32RoundMode_t roundingMode :
         {CAST_ROUND_HALF_NE, CAST_ROUND_DOWN, CAST_ROUND_UP, CAST_ROUND_ZERO, CAST_ROUND_DEFAULT, CAST_ROUND_HALF_AZ})
    {
        for (synDataType toDataType : {synDataType::syn_type_bf16,
                                       synDataType::syn_type_fp16,
                                       synDataType::syn_type_int8,
                                       synDataType::syn_type_uint8,
                                       synDataType::syn_type_int16,
                                       synDataType::syn_type_uint16,
                                       synDataType::syn_type_int32,
                                       synDataType::syn_type_uint32})
        {
            runCast<ns_CastKernel::ParamsV4>(synDataType::syn_type_float, toDataType, {2.5}, roundingMode);
            runCast<ns_CastKernel::ParamsV4>(synDataType::syn_type_float, toDataType, {1.66666666666}, roundingMode);
            runCast<ns_CastKernel::ParamsV4>(synDataType::syn_type_float, toDataType, {-0.1111111}, roundingMode);
            runCast<ns_CastKernel::ParamsV4>(synDataType::syn_type_float,
                                             toDataType,
                                             {std::numeric_limits<float>::min()},
                                             roundingMode);
            runCast<ns_CastKernel::ParamsV4>(synDataType::syn_type_float,
                                             toDataType,
                                             {std::numeric_limits<float>::max()},
                                             roundingMode);
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_fp16_to_all)
{
    for (CastF32RoundMode_t roundingMode :
         {CAST_ROUND_HALF_NE, CAST_ROUND_DOWN, CAST_ROUND_UP, CAST_ROUND_ZERO, CAST_ROUND_DEFAULT, CAST_ROUND_HALF_AZ})
    {
        for (synDataType toDataType : {synDataType::syn_type_float,
                                       synDataType::syn_type_bf16,
                                       synDataType::syn_type_int8,
                                       synDataType::syn_type_uint8,
                                       synDataType::syn_type_int16,
                                       synDataType::syn_type_uint16,
                                       synDataType::syn_type_int32,
                                       synDataType::syn_type_uint32})
        {
            runCast<ns_CastKernel::ParamsV3>(synDataType::syn_type_fp16, toDataType, {2.5}, roundingMode);
            runCast<ns_CastKernel::ParamsV3>(synDataType::syn_type_fp16, toDataType, {1.66666666666}, roundingMode);
            runCast<ns_CastKernel::ParamsV3>(synDataType::syn_type_fp16, toDataType, {-0.1111111}, roundingMode);
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_bf16_to_all)
{
    for (CastF32RoundMode_t roundingMode :
         {CAST_ROUND_HALF_NE, CAST_ROUND_DOWN, CAST_ROUND_UP, CAST_ROUND_ZERO, CAST_ROUND_DEFAULT, CAST_ROUND_HALF_AZ})
    {
        for (synDataType toDataType : {synDataType::syn_type_float,
                                       synDataType::syn_type_fp16,
                                       synDataType::syn_type_int8,
                                       synDataType::syn_type_uint8,
                                       synDataType::syn_type_int16,
                                       synDataType::syn_type_uint16,
                                       synDataType::syn_type_int32,
                                       synDataType::syn_type_uint32})
        {
            runCast<ns_CastKernel::ParamsV2>(synDataType::syn_type_bf16, toDataType, {2.5}, roundingMode);
            runCast<ns_CastKernel::ParamsV2>(synDataType::syn_type_bf16, toDataType, {1.66666666666}, roundingMode);
            runCast<ns_CastKernel::ParamsV2>(synDataType::syn_type_bf16, toDataType, {-0.1111111}, roundingMode);
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_int32_to_all)
{
    for (synDataType toDataType : {synDataType::syn_type_float,
                                   synDataType::syn_type_fp16,
                                   synDataType::syn_type_bf16,
                                   synDataType::syn_type_int8,
                                   synDataType::syn_type_uint8,
                                   synDataType::syn_type_int16,
                                   synDataType::syn_type_uint16,
                                   synDataType::syn_type_uint32})
    {
        runCast(synDataType::syn_type_int32, toDataType, getIntVal(1), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int32, toDataType, getIntVal(16384), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int32, toDataType, getIntVal(-1), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int32, toDataType, getIntVal(-1000), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int32, toDataType, {std::numeric_limits<int32_t>::min()}, CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int32, toDataType, {std::numeric_limits<int32_t>::max()}, CAST_ROUND_DEFAULT);
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_uint32_to_all)
{
    runCast(synDataType::syn_type_uint32, syn_type_int32, getIntVal(-10), CAST_ROUND_DEFAULT);

    for (synDataType toDataType : {synDataType::syn_type_float,
                                   synDataType::syn_type_fp16,
                                   synDataType::syn_type_bf16,
                                   synDataType::syn_type_int8,
                                   synDataType::syn_type_uint8,
                                   synDataType::syn_type_int16,
                                   synDataType::syn_type_uint16,
                                   synDataType::syn_type_int32})
    {
        runCast(synDataType::syn_type_uint32, toDataType, getIntVal(1), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint32, toDataType, getIntVal(16390), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint32, toDataType, getIntVal(-10), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint32, toDataType, getIntVal(-1000), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint32, toDataType, {std::numeric_limits<uint32_t>::max()}, CAST_ROUND_DEFAULT);
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_int16_to_all)
{
    for (synDataType toDataType : {synDataType::syn_type_float,
                                   synDataType::syn_type_fp16,
                                   synDataType::syn_type_bf16,
                                   synDataType::syn_type_int8,
                                   synDataType::syn_type_uint8,
                                   synDataType::syn_type_uint16,
                                   synDataType::syn_type_int32,
                                   synDataType::syn_type_uint32})
    {
        runCast(synDataType::syn_type_int16, toDataType, getIntVal(1), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int16, toDataType, getIntVal(256), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int16, toDataType, getIntVal(-1), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int16, toDataType, getIntVal(-1000), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int16, toDataType, {std::numeric_limits<int16_t>::min()}, CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int16, toDataType, {std::numeric_limits<int16_t>::max()}, CAST_ROUND_DEFAULT);
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_uint16_to_all)
{
    for (synDataType toDataType : {synDataType::syn_type_float,
                                   synDataType::syn_type_fp16,
                                   synDataType::syn_type_bf16,
                                   synDataType::syn_type_int8,
                                   synDataType::syn_type_uint8,
                                   synDataType::syn_type_int16,
                                   synDataType::syn_type_int32,
                                   synDataType::syn_type_uint32})
    {
        runCast(synDataType::syn_type_uint16, toDataType, getIntVal(5), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint16, toDataType, getIntVal(12950), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint16, toDataType, getIntVal(-1), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint16, toDataType, getIntVal(-1000), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint16, toDataType, {std::numeric_limits<uint16_t>::max()}, CAST_ROUND_DEFAULT);
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_int8_to_all)
{
    for (synDataType toDataType : {synDataType::syn_type_float,
                                   synDataType::syn_type_fp16,
                                   synDataType::syn_type_bf16,
                                   synDataType::syn_type_uint8,
                                   synDataType::syn_type_int16,
                                   synDataType::syn_type_uint16,
                                   synDataType::syn_type_int32,
                                   synDataType::syn_type_uint32})
    {
        runCast(synDataType::syn_type_int8, toDataType, getIntVal(9), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int8, toDataType, getIntVal(-9), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int8, toDataType, {std::numeric_limits<int8_t>::min()}, CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_int8, toDataType, {std::numeric_limits<int8_t>::max()}, CAST_ROUND_DEFAULT);
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_cast_coverage_uint8_to_all)
{
    for (synDataType toDataType : {synDataType::syn_type_float,
                                   synDataType::syn_type_fp16,
                                   synDataType::syn_type_bf16,
                                   synDataType::syn_type_int8,
                                   synDataType::syn_type_int16,
                                   synDataType::syn_type_uint16,
                                   synDataType::syn_type_int32,
                                   synDataType::syn_type_uint32})
    {
        runCast(synDataType::syn_type_uint8, toDataType, getIntVal(15), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint8, toDataType, getIntVal(-15), CAST_ROUND_DEFAULT);
        runCast(synDataType::syn_type_uint8, toDataType, {std::numeric_limits<uint8_t>::max()}, CAST_ROUND_DEFAULT);
    }
}

void SynTrainingTpcDualExecutionTest::runConstantCastAdd(synDataType fromDataType,
                                                         synDataType toDataType,
                                                         fint_t      constant)
{
    std::array<unsigned, 2> sizes         = {16, 16};
    std::array<unsigned, 1> constantSizes = {1};

    auto constantOut = createTensors(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     constantSizes.data(),
                                     constantSizes.size(),
                                     fromDataType);

    auto castOut = createTensors(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 constantSizes.data(),
                                 constantSizes.size(),
                                 toDataType);

    auto addOperand2 = createPersistTensors(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            sizes.data(),
                                            sizes.size(),
                                            toDataType);

    auto addOut =
        createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), sizes.size(), toDataType);
    ns_ConstantKernel::Params params             = {constant};
    TensorIndicesPair         constantOutIndices = {{constantOut.graph}, {constantOut.eager}};
    addNodesToGraphs(fmt::format("constant_{}", getDtypeSuffixFromSynDataType(fromDataType)).c_str(),
                     {},
                     constantOutIndices,
                     &params,
                     sizeof(params));

    TensorIndicesPair castInIndices  = {{constantOut.graph}, {constantOut.eager}};
    TensorIndicesPair castOutIndices = {{castOut.graph}, {castOut.eager}};
    addNodesToGraphs(fmt::format("cast_{}_to_{}",
                                 getDtypeSuffixFromSynDataType(fromDataType),
                                 getDtypeSuffixFromSynDataType(toDataType))
                         .c_str(),
                     castInIndices,
                     castOutIndices);

    TensorIndicesPair addInIndices  = {{castOut.graph, addOperand2.graph}, {castOut.eager, addOperand2.eager}};
    TensorIndicesPair addOutIndices = {{addOut.graph}, {addOut.eager}};
    addNodesToGraphs(fmt::format("add_fwd_{}", getDtypeSuffixFromSynDataType(toDataType)).c_str(),
                     addInIndices,
                     addOutIndices);

    compileAndRun();

    for (uint64_t i = 0; i < getNumberOfElements(sizes.data(), sizes.size()); i++)
    {
        if (isTypeFloat(toDataType))
        {
            float eagerVal = getBufferValue<float>(addOut.eager, i, toDataType);
            float graphVal = getBufferValue<float>(addOut.graph, i, toDataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
        else
        {
            int64_t eagerVal = getBufferValue<int64_t>(addOut.eager, i, toDataType);
            int64_t graphVal = getBufferValue<int64_t>(addOut.graph, i, toDataType);
            ASSERT_EQ(graphVal, eagerVal)
                << "Graph mode mismatch at index " << i << " Graph mode:" << graphVal << " Eager mode: " << eagerVal;
        }
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_fp32_to_bf16)
{
    runConstantCastAdd(synDataType::syn_type_float, synDataType::syn_type_bf16, {2.5});
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_bf16_to_fp32)
{
    runConstantCastAdd(synDataType::syn_type_bf16, synDataType::syn_type_float, {2.6});
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_fp32_to_fp16)
{
    runConstantCastAdd(synDataType::syn_type_float, synDataType::syn_type_fp16, {3.1});
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_fp16_to_fp32)
{
    runConstantCastAdd(synDataType::syn_type_fp16, synDataType::syn_type_float, {3.5});
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_fp32_to_i32)
{
    runConstantCastAdd(synDataType::syn_type_float, synDataType::syn_type_int32, {4.0});
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_i32_to_fp32)
{
    runConstantCastAdd(synDataType::syn_type_int32, synDataType::syn_type_float, getIntVal(4));
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_fp32_to_i16)
{
    runConstantCastAdd(synDataType::syn_type_float, synDataType::syn_type_int16, {4.0});
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_i16_to_fp32)
{
    runConstantCastAdd(synDataType::syn_type_int16, synDataType::syn_type_float, getIntVal(4));
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_fp32_to_i8)
{
    runConstantCastAdd(synDataType::syn_type_float, synDataType::syn_type_int8, {4.0});
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_constant_cast_add_i8_to_fp32)
{
    runConstantCastAdd(synDataType::syn_type_int8, synDataType::syn_type_float, getIntVal(4));
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_nll_loss)
{
    std::array<unsigned, 2> in1DimSizes = {24, 10};
    std::array<unsigned, 1> in2DimSizes = {10};
    std::array<unsigned, 1> outDimSizes = {1};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in1DimSizes.data(),
                         in1DimSizes.size(),
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in2DimSizes.data(),
                         in2DimSizes.size(),
                         syn_type_int32);
    auto          yTensorIndexPair      = createPersistTensors(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outDimSizes.data(),
                                                 outDimSizes.size(),
                                                 syn_type_single);
    unsigned char g_0_TPC463_0_params[] = {0, 0, 0, 0, -100, -1, -1, -1};

    addNodesToGraphs("nll_loss_fwd_f32", &g_0_TPC463_0_params, sizeof(g_0_TPC463_0_params));

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outDimSizes.data(), outDimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_bincount_i32)
{
    std::array<unsigned, 1> in1DimSizes = {128};
    std::array<unsigned, 1> in2DimSizes = {1};
    std::array<unsigned, 1> outDimSizes = {100};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in1DimSizes.data(),
                         in1DimSizes.size(),
                         syn_type_int32);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         in2DimSizes.data(),
                         in2DimSizes.size(),
                         syn_type_int32);
    auto yTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outDimSizes.data(),
                                                 outDimSizes.size(),
                                                 syn_type_int32);

    ns_BinCountKernel::Params bincount_params = {NO_WEIGHT};

    addNodesToGraphs("bincount_i32", &bincount_params, sizeof(bincount_params));

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(outDimSizes.data(), outDimSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_strided_view_add)
{
    std::array<unsigned, 2> stridedViewInputSizes  = {3, 3};
    std::array<unsigned, 2> stridedViewOutputSizes = {2, 2};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         stridedViewInputSizes.data(),
                         stridedViewInputSizes.size(),
                         syn_type_single);
    auto stridedViewOutPair = createTensors(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            stridedViewOutputSizes.data(),
                                            stridedViewOutputSizes.size(),
                                            syn_type_single);

    synStridedOpParams sv_params = {1, {2, 1}};
    addNodesToGraphs("strided_view", &sv_params, sizeof(sv_params));

    std::array<unsigned, 1> constantSizes = {1};

    auto constantOutPair = createTensors(OUTPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         constantSizes.data(),
                                         constantSizes.size(),
                                         syn_type_single);

    ns_ConstantKernel::Params constParams        = {1.0};
    TensorIndicesPair         constantOutIndices = {{constantOutPair.graph}, {constantOutPair.eager}};
    addNodesToGraphs("constant_f32", {}, {constantOutIndices}, &constParams, sizeof(constParams));

    std::array<unsigned, 2> addOutputSizes = {2, 2};

    auto addOutPair = createPersistTensors(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           addOutputSizes.data(),
                                           addOutputSizes.size(),
                                           syn_type_single);

    TensorIndicesPair addInIndices  = {{stridedViewOutPair.graph, constantOutPair.graph},
                                      {stridedViewOutPair.eager, constantOutPair.eager}};
    TensorIndicesPair addOutIndices = {{addOutPair.graph}, {addOutPair.eager}};

    addNodesToGraphs("add_fwd_f32", addInIndices, addOutIndices);

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[addOutPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[addOutPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(addOutputSizes.data(), addOutputSizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTpcDualExecutionTest, basic_frobenius_norm)
{
    std::array<unsigned, 4> fMsizes     = {64, 8, 4, 1};
    std::array<unsigned, 2> oneDimSizes = {1};

    createPersistTensors(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, fMsizes.data(), 4, syn_type_float);

    auto scalarOutPair =
        createPersistTensors(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oneDimSizes.data(), 1, syn_type_float);

    addNodesToGraphs("frobenius_norm_fwd");

    compileAndRun();

    auto pOutputScalarGraphMode = *static_cast<float*>(m_hostBuffers[scalarOutPair.graph]);
    auto pOutputScalarEagerMode = *static_cast<float*>(m_hostBuffers[scalarOutPair.eager]);
    ASSERT_EQ(pOutputScalarGraphMode, pOutputScalarEagerMode)
        << "mismatch Graph mode:" << pOutputScalarGraphMode << " Eager mode: " << pOutputScalarEagerMode;
}