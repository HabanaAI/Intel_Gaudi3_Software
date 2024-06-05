#include "data_type_utils.h"
#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "test_utils.h"
#include <habana_global_conf.h>
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdint>

using SynDataTypePair = std::pair<synDataType, synDataType>;

class SynGaudiCast64BitTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<SynDataTypePair, TestSizeVec, TestSizeVec>>
/* Test Parameters: Cast I/O syn data type pair, Sizes vector */
{
public:
    SynGaudiCast64BitTest()
    : m_inDtype(std::get<0>(GetParam()).first),
      m_outDtype(std::get<0>(GetParam()).second),
      m_inSizes(std::get<1>(GetParam())),
      m_outSizes(m_inSizes),
      m_dim(static_cast<unsigned>(m_inSizes.size())),
      m_nElements(getNumberOfElements(m_inSizes.data(), m_inSizes.size()))
    {
    }

    static void validateCast(synDataType inType, unsigned nElem, void* pInTensor, void* pOutTensor);
    template<typename InDtype, typename OutDtype>
    static void cast64BitValidator(InDtype inTensor[], OutDtype outTensor[], uint64_t nElem);

protected:
    void run();

    const synDataType m_inDtype;
    const synDataType m_outDtype;
    TestSizeVec       m_inSizes;
    TestSizeVec       m_outSizes;
    const unsigned    m_dim;
    unsigned          m_inTensorIdx;
    unsigned          m_outTensorIdx;
    float*            m_pInputData {nullptr};
    uint64_t          m_nElements;
};

template<typename InDtype, typename OutDtype>
void SynGaudiCast64BitTest::cast64BitValidator(InDtype inTensor[], OutDtype outTensor[], uint64_t nElem)
{
    for (auto idx = 0; idx < nElem; ++idx)
    {
        ASSERT_EQ(inTensor[idx], static_cast<InDtype>(outTensor[idx])) << " [idx=" << idx << "]";
    }
}

void SynGaudiCast64BitTest::validateCast(synDataType inType, unsigned nElem, void* pInTensor, void* pOutTensor)
{
    switch (inType)
    {
        case syn_type_int64:
            cast64BitValidator<int64_t, int32_t>(static_cast<int64_t*>(pInTensor),
                                                 static_cast<int32_t*>(pOutTensor),
                                                 nElem);
            break;
        case syn_type_uint64:
            cast64BitValidator<uint64_t, uint32_t>(static_cast<uint64_t*>(pInTensor),
                                                   static_cast<uint32_t*>(pOutTensor),
                                                   nElem);
            break;
        case syn_type_int32:
            cast64BitValidator<int32_t, int64_t>(static_cast<int32_t*>(pInTensor),
                                                 static_cast<int64_t*>(pOutTensor),
                                                 nElem);
            break;
        case syn_type_uint32:
            cast64BitValidator<uint32_t, uint64_t>(static_cast<uint32_t*>(pInTensor),
                                                   static_cast<uint64_t*>(pOutTensor),
                                                   nElem);
            break;

        default:
            EXPECT_TRUE(false) << "Invalid tensor type";
            break;
    }
}

void SynGaudiCast64BitTest::run()
{
    m_inTensorIdx = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_POSITIVE,
                                        nullptr,
                                        m_inSizes.data(),
                                        m_inSizes.size(),
                                        m_inDtype,
                                        nullptr,
                                        "input");

    m_outTensorIdx = createPersistTensor(OUTPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         m_outSizes.data(),
                                         m_outSizes.size(),
                                         m_outDtype,
                                         nullptr,
                                         "output");

    auto&& castGuid = getCastGUID(m_inDtype, m_outDtype);
    addNodeToGraph(castGuid.c_str(), {m_inTensorIdx}, {m_outTensorIdx}, nullptr, 0, "cast");

    compileAndRun();

    validateCast(m_inDtype, m_nElements, m_hostBuffers[m_inTensorIdx], m_hostBuffers[m_outTensorIdx]);
}

class SynGaudiCast64BitDynamicTest : public SynGaudiCast64BitTest
/* Test Parameters: Cast I/O syn data type pair, Sizes vector */
{
public:
    SynGaudiCast64BitDynamicTest() : SynGaudiCast64BitTest(), m_inSizesMin(std::get<2>(GetParam()))
    {
        setTestPackage(TEST_PACKAGE_DSD);
        m_nElements = getNumberOfElements(m_inSizesMin.data(), m_inSizesMin.size());
    }

protected:
    void run();

    TestSizeVec m_inSizesMin;
};

void SynGaudiCast64BitDynamicTest::run()
{
    m_inTensorIdx = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_POSITIVE,
                                        nullptr,
                                        m_inSizes.data(),
                                        m_inSizes.size(),
                                        m_inDtype,
                                        nullptr,
                                        "input",
                                        0,
                                        0,
                                        nullptr,
                                        m_inSizesMin.data());

    m_outTensorIdx = createPersistTensor(OUTPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         m_outSizes.data(),
                                         m_outSizes.size(),
                                         m_outDtype,
                                         nullptr,
                                         "output",
                                         0,
                                         0,
                                         nullptr,
                                         m_inSizesMin.data());

    auto&& castGuid = getCastGUID(m_inDtype, m_outDtype);
    addNodeToGraph(castGuid.c_str(), {m_inTensorIdx}, {m_outTensorIdx}, nullptr, 0, "cast");

    compileTopology();
    setActualSizes(m_inTensorIdx, m_inSizesMin);
    setActualSizes(m_outTensorIdx, m_inSizesMin);
    runTopology();

    validateCast(m_inDtype, m_nElements, m_hostBuffers[m_inTensorIdx], m_hostBuffers[m_outTensorIdx]);
}

TEST_P_GC(SynGaudiCast64BitTest, gaudi_64b_cast_tests, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    run();
}
TEST_P_GC(SynGaudiCast64BitDynamicTest, gaudi_64b_cast_dynamic_tests, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    run();
}

INSTANTIATE_TEST_SUITE_P(downcast_static_shape,
                         SynGaudiCast64BitTest,
                         ::testing::Combine(::testing::ValuesIn({SynDataTypePair(syn_type_int64, syn_type_int32),
                                                                 SynDataTypePair(syn_type_uint64, syn_type_uint32)}),
                                            ::testing::ValuesIn({TestSizeVec {3, 3, 3, 2, 3}}),
                                            ::testing::ValuesIn({TestSizeVec {3, 3, 3, 2, 3}})));

INSTANTIATE_TEST_SUITE_P(upcast_static_shape,
                         SynGaudiCast64BitTest,
                         ::testing::Combine(::testing::ValuesIn({SynDataTypePair(syn_type_int32, syn_type_int64),
                                                                 SynDataTypePair(syn_type_uint32, syn_type_uint64)}),
                                            ::testing::ValuesIn({TestSizeVec {3, 3, 3, 2, 3}}),
                                            ::testing::ValuesIn({TestSizeVec {3, 3, 3, 2, 3}})));

INSTANTIATE_TEST_SUITE_P(downcast_shape_single_dim,
                         SynGaudiCast64BitDynamicTest,
                         ::testing::Combine(::testing::ValuesIn({SynDataTypePair(syn_type_int64, syn_type_int32),
                                                                 SynDataTypePair(syn_type_uint64, syn_type_uint32)}),
                                            ::testing::ValuesIn({TestSizeVec {7}}),
                                            ::testing::ValuesIn({TestSizeVec {3}})));

INSTANTIATE_TEST_SUITE_P(downcast,
                         SynGaudiCast64BitDynamicTest,
                         ::testing::Combine(::testing::ValuesIn({SynDataTypePair(syn_type_int64, syn_type_int32),
                                                                 SynDataTypePair(syn_type_uint64, syn_type_uint32)}),
                                            ::testing::ValuesIn({TestSizeVec {3, 3, 3, 2, 3}}),
                                            ::testing::ValuesIn({TestSizeVec {1, 3, 2, 2, 3}})));

INSTANTIATE_TEST_SUITE_P(upcast_dynamic_shape_single_dim,
                         SynGaudiCast64BitDynamicTest,
                         ::testing::Combine(::testing::ValuesIn({SynDataTypePair(syn_type_int32, syn_type_int64),
                                                                 SynDataTypePair(syn_type_uint32, syn_type_uint64)}),
                                            ::testing::ValuesIn({TestSizeVec {7}}),
                                            ::testing::ValuesIn({TestSizeVec {3}})));

INSTANTIATE_TEST_SUITE_P(upcast,
                         SynGaudiCast64BitDynamicTest,
                         ::testing::Combine(::testing::ValuesIn({SynDataTypePair(syn_type_int32, syn_type_int64),
                                                                 SynDataTypePair(syn_type_uint32, syn_type_uint64)}),
                                            ::testing::ValuesIn({TestSizeVec {3, 3, 3, 2, 3}}),
                                            ::testing::ValuesIn({TestSizeVec {1, 3, 2, 2, 3}})));

TEST_F_GC(SynGaudiTestInfra, int64_reshape)
{
    TestSizes            sizes           = {4, 3, 2, 3};
    auto                 dim             = 4u;
    auto                 castInTensorIdx = 0u;
    std::vector<int64_t> inputData;
    auto                 nElements = getNumberOfElements(sizes.data(), dim);
    inputData.resize(nElements);

    std::iota(inputData.begin(), inputData.end(), 1);
    castInTensorIdx = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_FROM_INITIALIZER,
                                          reinterpret_cast<float*>(inputData.data()),
                                          sizes.data(),
                                          dim,
                                          syn_type_int64,
                                          nullptr,
                                          "downcast_input");

    auto castOutTensorIdx = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                sizes.data(),
                                                dim,
                                                syn_type_int32,
                                                nullptr,
                                                "downcast_output");

    auto&& castGuid = getCastGUID(syn_type_int64, syn_type_int32);
    addNodeToGraph(castGuid.c_str(), {castInTensorIdx}, {castOutTensorIdx}, nullptr, 0, "downcast");

    std::array<unsigned, SYN_MAX_TENSOR_DIM> newSizes            = {12, 6};
    auto                                     newDim              = 2u;
    auto                                     reshapeOutTensorIdx = createPersistTensor(OUTPUT_TENSOR,
                                                                                       MEM_INIT_ALL_ZERO,
                                                                                       nullptr,
                                                                                       newSizes.data(),
                                                                                       newDim,
                                                                                       syn_type_int32,
                                                                                       nullptr,
                                                                                       "reshape_output");

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {castOutTensorIdx}, {reshapeOutTensorIdx}, nullptr, 0, "reshape");

    std::array<unsigned, SYN_MAX_TENSOR_DIM> splitOutSize = {6, 6};

    auto splitOutTensorIdx0 = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  splitOutSize.data(),
                                                  newDim,
                                                  syn_type_int32,
                                                  nullptr,
                                                  "split_output0");

    auto splitOutTensorIdx1 = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  splitOutSize.data(),
                                                  newDim,
                                                  syn_type_int32,
                                                  nullptr,
                                                  "split_output1");

    synSplitParams params = {.axis = 0 /* FCD */};
    addNodeToGraph(NodeFactory::splitNodeTypeName,
                   {reshapeOutTensorIdx},
                   {splitOutTensorIdx0, splitOutTensorIdx1},
                   &params,
                   sizeof(params),
                   "split");

    auto reshapeUpcastOutIdx = createPersistTensor(OUTPUT_TENSOR,
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   newSizes.data(),
                                                   newDim,
                                                   syn_type_int64,
                                                   nullptr,
                                                   "upcast_output");

    castGuid = getCastGUID(syn_type_int32, syn_type_int64);
    addNodeToGraph(castGuid.c_str(), {reshapeOutTensorIdx}, {reshapeUpcastOutIdx}, nullptr, 0, "upcast");

    compileAndRun();

    auto pSplitOutData0 = reinterpret_cast<int32_t*>(m_hostBuffers[splitOutTensorIdx0]);
    auto pSplitOutData1 = reinterpret_cast<int32_t*>(m_hostBuffers[splitOutTensorIdx1]);

    for (auto j = 0; j < newSizes[1]; ++j)
    {
        for (auto i = 0; i < newSizes[0]; ++i)
        {
            auto     idx                 = i + (j * newSizes[0]);
            auto     splitIdx            = 0u;
            int32_t* pSplitOutTensorBuff = nullptr;
            if (i < splitOutSize[0])
            {
                splitIdx            = i + (j * splitOutSize[0]);
                pSplitOutTensorBuff = pSplitOutData0;
            }
            else
            {
                splitIdx            = (i - splitOutSize[0]) + (j * splitOutSize[0]);
                pSplitOutTensorBuff = pSplitOutData1;
            }
            ASSERT_EQ(static_cast<int32_t*>(m_hostBuffers[reshapeOutTensorIdx])[idx], pSplitOutTensorBuff[splitIdx]);
        }
    }

    SynGaudiCast64BitTest::validateCast(syn_type_int32,
                                        nElements,
                                        m_hostBuffers[reshapeOutTensorIdx],
                                        m_hostBuffers[reshapeUpcastOutIdx]);
}
