#include "gtest/gtest-typed-test.h"
#include "gtest/gtest.h"
#include <stdint.h>
#include <memory>
#include "data_type_utils.h"
#include "node_operations_test.h"
#include "../gaudi_tests/dma_node_test.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "supported_devices_macros.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "mme_reference/data_types/non_standard_dtypes.h"

/****************************************************************************
****************************      MME TESTS      ****************************
*****************************************************************************/

class SynGaudi2ConvDataTypes
: public SynTrainingNodeOperations
, public testing::WithParamInterface<std::tuple<ERepefenceOp /* op */, synDataType /* data type */>>
{
};

INSTANTIATE_TEST_SUITE_P(
    MmeDataTypesTesting,
    SynGaudi2ConvDataTypes,
    ::testing::Combine(::testing::ValuesIn({REFERENCE_OP_FWD, REFERENCE_OP_DEDX, REFERENCE_OP_DEDW}),  // op
                       ::testing::ValuesIn({syn_type_single,
                                            syn_type_bf16,
                                            syn_type_fp16,
                                            syn_type_fp8_152,
                                            /*TODO SW-55163:
                                            syn_type_tf32,
                                            syn_type_hb_float*/})));  // synDataType

TEST_P_GC(SynGaudi2ConvDataTypes, conv_fwd_bwd, {synDeviceGaudi2})
{
    synConvolutionParams convParams;
    convParams.kH               = 3;
    convParams.kW               = 3;
    unsigned     ifmC           = 12;
    unsigned     ifmSpatialSize = 64;
    unsigned     batch          = 1;
    unsigned     ofmK           = 7;
    ERepefenceOp op             = std::get<0>(GetParam());
    synDataType  dtype          = std::get<1>(GetParam());

    TestSizes xSize = {ifmC, ifmSpatialSize, ifmSpatialSize, batch, 1};
    TestSizes wSize = {ofmK, ifmC, convParams.kW, convParams.kH, 1};
    TestSizes ySize = {
        ofmK,
        convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
        convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
        batch,
        1};

    // In case of dedw with bf16 or fp16 output, use tensor-global checking instead of element-wise, because cd concurrency
    // might be applied, for which specific elements may cross the element-wise threshold
    bool usePearsonCompare = (op == REFERENCE_OP_DEDW) &&
                             ((dtype == syn_type_fp16) || (dtype == syn_type_bf16));
    SynTrainingNodeOperations::runMmeTest(xSize, wSize, ySize, convParams, op, dtype, dtype, usePearsonCompare);
}

class SynGaudi2BGemmDataTypes
: public SynTrainingBatchGemmTest
, public testing::WithParamInterface<std::tuple<ERepefenceOp /* op */, synDataType /* data type */>>
{
};

INSTANTIATE_TEST_SUITE_P(
    MmeDataTypesTesting,
    SynGaudi2BGemmDataTypes,
    ::testing::Combine(::testing::ValuesIn({REFERENCE_OP_FWD, REFERENCE_OP_DEDX, REFERENCE_OP_DEDW}),  // op
                       ::testing::ValuesIn({syn_type_single,
                                            syn_type_bf16,
                                            syn_type_fp16,
                                            syn_type_fp8_152,
                                            /*TODO SW-55163:
                                            syn_type_tf32,
                                            syn_type_hb_float*/})));  // synDataType

TEST_P_GC(SynGaudi2BGemmDataTypes, bgemm_fwd_bwd)
{
    ERepefenceOp op    = std::get<0>(GetParam());
    synDataType  dtype = std::get<1>(GetParam());
    SynTrainingBatchGemmTest::doBatchGemmTest(TestSizes({128, 128, 1, 1, 1}),
                                              TestSizes({64, 128, 1, 1, 1}),
                                              4,
                                              op,
                                              dtype,
                                              dtype,
                                              nullptr);
}

class SynGaudi2GemmDataTypes
: public SynTrainingGemmTest
, public testing::WithParamInterface<std::tuple<synDataType /* data type */>>
{
};

INSTANTIATE_TEST_SUITE_P(MmeDataTypesTesting,
                         SynGaudi2GemmDataTypes,
                         ::testing::Combine(::testing::ValuesIn({syn_type_single,
                                                                 syn_type_bf16,
                                                                 syn_type_fp16,
                                                                 syn_type_fp8_152,
                                                                 /*TODO SW-55163:
                                                                 syn_type_tf32,
                                                                 syn_type_hb_float*/})));  // synDataType

TEST_P_GC(SynGaudi2GemmDataTypes, gemm_fwd)
{
    synDataType dtype = std::get<0>(GetParam());
    SynTrainingGemmTest::doGemmTest(std::array<unsigned, 2>({128, 256}),
                                    std::array<unsigned, 2>({512, 128}),
                                    false,
                                    false,
                                    dtype);
}

/****************************************************************************
****************************      TPC TESTS      ****************************
*****************************************************************************/

template<class T>
class SynGaudi2TpcDataTypes : public SynGaudiTestInfra
{
};

using testing::Types;
typedef Types<float, bfloat16> supportedTpcTypes;

TYPED_TEST_SUITE(SynGaudi2TpcDataTypes, supportedTpcTypes);

TYPED_TEST_GC(SynGaudi2TpcDataTypes, cast_f8_to, {synDeviceGaudi2})
{
    unsigned    sizes[]  = {4, 4, 1, 1};
    unsigned    dims     = 2;
    synDataType dtype    = dataTypeToSynType<TypeParam>();
    uint64_t    numElems = SynGaudiTestInfra::getNumberOfElements(sizes, dims);

    unsigned castIn  = SynGaudiTestInfra::createPersistTensor(INPUT_TENSOR,
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             sizes,
                                                             dims,
                                                             syn_type_fp8_152,
                                                             nullptr,
                                                             "castIn");
    unsigned castOut = SynGaudiTestInfra::createPersistTensor(OUTPUT_TENSOR,
                                                              MEM_INIT_ALL_ZERO,
                                                              nullptr,
                                                              sizes,
                                                              dims,
                                                              dtype,
                                                              nullptr,
                                                              "castOut");

    const std::string guidStr = fmt::format("cast_f8_to_{}", getDtypeSuffixFromSynDataType(dtype));
    SynGaudiTestInfra::addNodeToGraph(guidStr.c_str(), {castIn}, {castOut});

    SynGaudiTestInfra::compileAndRun();

    auto* input  = SynGaudiTestInfra::castHostBuffer<fp8_152_t>(castIn);
    auto* output = SynGaudiTestInfra::castHostBuffer<TypeParam>(castOut);

    for (uint64_t idx = 0; idx < numElems; idx++)
    {
        ASSERT_FLOAT_EQ(input[idx].toFloat(EXPONENT_BIAS_FP8_152_15), (float)output[idx])
            << "OUTPUT: Mismatch for at index " << idx;
    }
}

TYPED_TEST_GC(SynGaudi2TpcDataTypes, cast_to_f8, {synDeviceGaudi2})
{
    unsigned    sizes[]  = {4, 4, 1, 1};
    unsigned    dims     = 2;
    uint64_t    numElems = SynGaudiTestInfra::getNumberOfElements(sizes, dims);
    synDataType dtype    = dataTypeToSynType<TypeParam>();

    unsigned castIn      = SynGaudiTestInfra::createPersistTensor(INPUT_TENSOR,
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             sizes,
                                                             dims,
                                                             dtype,
                                                             nullptr,
                                                             "castIn");
    unsigned castOut     = SynGaudiTestInfra::createPersistTensor(OUTPUT_TENSOR,
                                                              MEM_INIT_ALL_ZERO,
                                                              nullptr,
                                                              sizes,
                                                              dims,
                                                              syn_type_fp8_152,
                                                              nullptr,
                                                              "castOut");
    const std::string guidStr     = fmt::format("cast_{}_to_f8", getDtypeSuffixFromSynDataType(dtype));
    SynGaudiTestInfra::addNodeToGraph(guidStr.c_str(), {castIn}, {castOut});

    SynGaudiTestInfra::compileAndRun();

    auto* input  = SynGaudiTestInfra::castHostBuffer<TypeParam>(castIn);
    auto* output = SynGaudiTestInfra::castHostBuffer<fp8_152_t>(castOut);

    for (uint64_t idx = 0; idx < numElems; idx++)
    {
        auto expected = fp8_152_t(float(input[idx]), MmeCommon::RoundToNearest);
        ASSERT_FLOAT_EQ(expected.toFloat(EXPONENT_BIAS_FP8_152_15), output[idx].toFloat(EXPONENT_BIAS_FP8_152_15))
            << "OUTPUT: Mismatch for at index " << idx;
    }
}

/****************************************************************************
****************************      DMA TESTS      ****************************
*****************************************************************************/

using testing::Types;
typedef Types<fp16_t, float, bfloat16, fp8_152_t, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t> supportedTypes;

using testing::Types;
typedef Types<fp16_t, float, bfloat16, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t>
    supportedTypesDmaTranspose;

template<class T>
class SynGaudi2DmaTransposeDataTypes : public SynGaudiTestInfra
{
};

TYPED_TEST_SUITE(SynGaudi2DmaTransposeDataTypes, supportedTypesDmaTranspose);

TYPED_TEST_GC(SynGaudi2DmaTransposeDataTypes, gaudi2_dma_transpose, {synDeviceGaudi2})
{
    GlobalConfTestSetter conf("ENABLE_INTERNAL_NODES", "true");
    synDataType          dtype     = dataTypeToSynType<TypeParam>();
    unsigned             dims      = 2;
    unsigned             inSize[]  = {64, 16, 1, 1, 1};
    unsigned             outSize[] = {16, 64, 1, 1, 1};

    auto in  = SynGaudiTestInfra::createPersistTensor(INPUT_TENSOR,
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     inSize,
                                                     dims,
                                                     dtype);
    auto out = SynGaudiTestInfra::createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSize, dims, dtype);
    SynGaudiTestInfra::addNodeToGraph(NodeFactory::transposeDmaNodeTypeName, {in}, {out});

    SynGaudiTestInfra::compileAndRun();

    auto* dmaInput  = (TypeParam*)SynGaudiTestInfra::m_hostBuffers[in];
    auto* dmaOutput = (TypeParam*)SynGaudiTestInfra::m_hostBuffers[out];

    for (int i = 0; i < inSize[1]; ++i)
    {
        for (int j = 0; j < inSize[0]; ++j)
        {
            auto inIdx  = i * inSize[0] + j;
            auto outIdx = j * inSize[1] + i;
            ASSERT_EQ(dmaInput[inIdx], dmaOutput[outIdx]);
        }
    }
}

template<class T>
class SynGaudi2DmaMemcpyDataTypes : public SynGaudiTestDma
{
};

TYPED_TEST_SUITE(SynGaudi2DmaMemcpyDataTypes, supportedTypes);

TYPED_TEST_GC(SynGaudi2DmaMemcpyDataTypes, gaudi2_linear_dma_node, {synDeviceGaudi2})
{
    SynGaudiTestDma::linear_dma_node<TypeParam>();
}

TYPED_TEST_GC(SynGaudi2DmaMemcpyDataTypes, gaudi2_strided_dma_node, {synDeviceGaudi2})
{
    SynGaudiTestDma::strided_dma_node<TypeParam>();
}

TYPED_TEST_GC(SynGaudi2DmaMemcpyDataTypes, gaudi2_three_dimensional_strided_dma_node, {synDeviceGaudi2})
{
    SynGaudiTestDma::three_dimensional_strided_dma_node<TypeParam>();
}

template<class T>
class SynGaudi2DmaMemsetDataTypes : public SynGaudiTestDmaMemset
{
};

TYPED_TEST_SUITE(SynGaudi2DmaMemsetDataTypes, supportedTypes);

TYPED_TEST_GC(SynGaudi2DmaMemsetDataTypes, linear_memset_dma_node, {synDeviceGaudi2})
{
    SynGaudiTestDmaMemset::linear_memset_dma_node<TypeParam>();
}

TYPED_TEST_GC(SynGaudi2DmaMemsetDataTypes, linear_memset_2d_dma_node, {synDeviceGaudi2})
{
    SynGaudiTestDmaMemset::linear_memset_2d_dma_node<TypeParam>();
}

TYPED_TEST_GC(SynGaudi2DmaMemsetDataTypes, three_dimensional_memset_node, {synDeviceGaudi2})
{
    SynGaudiTestDmaMemset::three_dimensional_memset_node<TypeParam>();
}

/****************************************************************************
**************************      GC OPS TESTS      ***************************
*****************************************************************************/

template<class T>
class SynGaudi2LogicalSplitConcatDataTypes : public SynGaudiTestInfra
{
};
TYPED_TEST_SUITE(SynGaudi2LogicalSplitConcatDataTypes, supportedTypes);

TYPED_TEST_GC(SynGaudi2LogicalSplitConcatDataTypes, split_concat, {synDeviceGaudi2})
{
    unsigned    inSizes[]           = {384, 40, 15};
    unsigned    dims                = 3;
    unsigned    intermediateSizes[] = {128, 40, 15};
    synDataType dtype               = dataTypeToSynType<TypeParam>();

    unsigned inTensor = SynGaudiTestInfra::createPersistTensor(INPUT_TENSOR,
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               inSizes,
                                                               dims,
                                                               dtype,
                                                               nullptr,
                                                               "splitIn");

    unsigned intermediateTensor1 = SynGaudiTestInfra::createTensor(OUTPUT_TENSOR,
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   intermediateSizes,
                                                                   dims,
                                                                   dtype);
    unsigned intermediateTensor2 = SynGaudiTestInfra::createTensor(OUTPUT_TENSOR,
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   intermediateSizes,
                                                                   dims,
                                                                   dtype);
    unsigned intermediateTensor3 = SynGaudiTestInfra::createTensor(OUTPUT_TENSOR,
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   intermediateSizes,
                                                                   dims,
                                                                   dtype);

    unsigned outTensor = SynGaudiTestInfra::createPersistTensor(OUTPUT_TENSOR,
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                inSizes,
                                                                dims,
                                                                dtype,
                                                                nullptr,
                                                                "ConcatOut");

    synSplitParams splitParams = {0};

    SynGaudiTestInfra::addNodeToGraph(NodeFactory::splitNodeTypeName,
                                      {inTensor},
                                      {intermediateTensor1, intermediateTensor2, intermediateTensor3},
                                      &splitParams,
                                      sizeof(splitParams));

    synConcatenateParams concatenateParams = {0};

    SynGaudiTestInfra::addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                                      {intermediateTensor1, intermediateTensor2, intermediateTensor3},
                                      {outTensor},
                                      &concatenateParams,
                                      sizeof(concatenateParams));

    SynGaudiTestInfra::compileAndRun();

    auto* inData          = SynGaudiTestInfra::castHostInBuffer<TypeParam>(inTensor);
    auto* outData         = SynGaudiTestInfra::castHostOutBuffer<TypeParam>(outTensor);
    auto  totalActualSize = inSizes[0] * inSizes[1] * inSizes[2];

    for (int i = 0; i < totalActualSize; i++)
    {
        ASSERT_EQ(inData[i], outData[i]);
    }
}
