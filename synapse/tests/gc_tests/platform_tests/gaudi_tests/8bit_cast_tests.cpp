#include "data_type_utils.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "test_utils.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"

using CastTypeRelation = std::pair<synDataType, synDataType>;

class SynGaudiCast8bTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<CastTypeRelation, TestSizeVec>>
/* Test Parameters: Cast I/O syn data type pair, Sizes vector */
{
public:
    SynGaudiCast8bTest()
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
    static void castValidator(InDtype inTensor[], OutDtype outTensor[], uint64_t nElem);

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
    const uint32_t    m_nElements;
};

template<typename InDtype, typename OutDtype>
void SynGaudiCast8bTest::castValidator(InDtype inTensor[], OutDtype outTensor[], uint64_t nElem)
{
    for (auto idx = 0; idx < nElem; ++idx)
    {
        ASSERT_EQ(static_cast<float>(inTensor[idx]), static_cast<float>(outTensor[idx]));
    }
}

void SynGaudiCast8bTest::validateCast(synDataType inType, unsigned nElem, void* pInTensor, void* pOutTensor)
{
    switch (inType)
    {
        case syn_type_int8:
            castValidator<int8_t, int32_t>(static_cast<int8_t*>(pInTensor), static_cast<int32_t*>(pOutTensor), nElem);
            break;
        case syn_type_uint8:
            castValidator<uint8_t, uint32_t>(static_cast<uint8_t*>(pInTensor),
                                             static_cast<uint32_t*>(pOutTensor),
                                             nElem);
            break;
        case syn_type_fp8_152:
            castValidator<fp8_152_t, bfloat16>(static_cast<fp8_152_t*>(pInTensor),
                                               static_cast<bfloat16*>(pOutTensor),
                                               nElem);
            break;
        case syn_type_int32:
            castValidator<int32_t, int8_t>(static_cast<int32_t*>(pInTensor), static_cast<int8_t*>(pOutTensor), nElem);
            break;
        case syn_type_uint32:
            castValidator<uint32_t, uint8_t>(static_cast<uint32_t*>(pInTensor),
                                             static_cast<uint8_t*>(pOutTensor),
                                             nElem);
            break;
        case syn_type_bf16:
            validateResult(static_cast<bfloat16*>(pInTensor), static_cast<fp8_152_t*>(pOutTensor), nElem);
            break;

        default:
            EXPECT_TRUE(false) << "Invalid tensor type";
            break;
    }
}

void SynGaudiCast8bTest::run()
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

    SynGaudiCast8bTest::validateCast(m_inDtype,
                                     m_nElements,
                                     m_hostBuffers[m_inTensorIdx],
                                     m_hostBuffers[m_outTensorIdx]);
}

TEST_P_GC(SynGaudiCast8bTest, gaudi_8b_cast_tests, {synDeviceGaudi2})
{
    run();
}

INSTANTIATE_TEST_SUITE_P(downcast,
                         SynGaudiCast8bTest,
                         ::testing::Combine(::testing::ValuesIn({CastTypeRelation(syn_type_int8, syn_type_int32),
                                                                 CastTypeRelation(syn_type_uint8, syn_type_uint32),
                                                                 CastTypeRelation(syn_type_fp8_152, syn_type_bf16)}),
                                            ::testing::ValuesIn({TestSizeVec {5}, TestSizeVec {2, 3, 3, 2, 3}})));

INSTANTIATE_TEST_SUITE_P(upcast,
                         SynGaudiCast8bTest,
                         ::testing::Combine(::testing::ValuesIn({CastTypeRelation(syn_type_int32, syn_type_int8),
                                                                 CastTypeRelation(syn_type_uint32, syn_type_uint8),
                                                                 CastTypeRelation(syn_type_bf16, syn_type_fp8_152)}),
                                            ::testing::ValuesIn({TestSizeVec {5}, TestSizeVec {2, 3, 3, 2, 3}})));