#include "data_type_utils.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "syn_data_type_type_conversions.h"
#include "synapse_common_types.h"
#include "gtest/gtest.h"
#include "scoped_configuration_change.h"

class SynGaudi64BitMemoryOpTests : public SynGaudiTestInfra
{
public:
    static constexpr auto mask64bDataType = (syn_type_uint64 | syn_type_int64);

protected:
    void validateMemOp(unsigned lhsIdx, unsigned rhsIdx, uint64_t nElem, synDataType dtype)
    {
        switch (dtype)
        {
            case syn_type_int64:
                validateTensorsData<AsCppType<syn_type_int64>>(lhsIdx, rhsIdx, nElem);
                break;
            case syn_type_uint64:
                validateTensorsData<AsCppType<syn_type_uint64>>(lhsIdx, rhsIdx, nElem);
                break;
            case syn_type_int32:
                validateTensorsData<AsCppType<syn_type_int32>>(lhsIdx, rhsIdx, nElem);
                break;
            case syn_type_uint32:
                validateTensorsData<AsCppType<syn_type_uint32>>(lhsIdx, rhsIdx, nElem);
                break;
            default:
                EXPECT_FALSE(true);
                break;
        }
    }

    template<typename T>
    void validateTensorsData(unsigned lhsIdx, unsigned rhsIdx, uint64_t nElem)
    {
        const T* pLhsData = castHostBuffer<T>(lhsIdx);
        const T* pRhsData = castHostBuffer<T>(rhsIdx);
        for (uint32_t idx = 0; idx < nElem; ++idx, ++pLhsData, ++pRhsData)
        {
            ASSERT_EQ(*pLhsData, *pRhsData) << "Tensor data comparison failed";
        }
    }

private:
};

/**
 * Test Parameters:
 * ----------------
 * (0) syn data type
 * (1) sizes
 */
class SynGaudi64BitMemcpyTests
: public SynGaudi64BitMemoryOpTests
, public ::testing::WithParamInterface<std::tuple<synDataType, TestSizeVec>>
{
public:
protected:
};

TEST_P_GC(SynGaudi64BitMemcpyTests, 64b_memcpy)
{
    const auto& dtype = std::get<0>(GetParam());
    const auto& sizes = std::get<1>(GetParam());
    const auto  dim   = sizes.size();

    EXPECT_TRUE((dtype & mask64bDataType) != 0) << "Input type isn't 64b";

    auto memcpyIn = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        const_cast<unsigned*>(sizes.data()),
                                        dim,
                                        dtype,
                                        nullptr,
                                        "memcpy_in");

    auto memcpyOut = createPersistTensor(OUTPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         const_cast<unsigned*>(sizes.data()),
                                         dim,
                                         dtype,
                                         nullptr,
                                         "memcpy_out");

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {memcpyIn}, {memcpyOut}, nullptr, 0, "memcpy");

    compileAndRun();

    const auto nElem = getNumberOfElements(sizes.data(), dim);
    validateMemOp(memcpyIn, memcpyOut, nElem, dtype);
}

INSTANTIATE_TEST_SUITE_P(low_rank,
                         SynGaudi64BitMemcpyTests,
                         ::testing::Combine(::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
                                            ::testing::Values(TestSizeVec {5}, TestSizeVec {2, 4, 3})));

INSTANTIATE_TEST_SUITE_P(high_rank,
                         SynGaudi64BitMemcpyTests,
                         ::testing::Combine(::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
                                            ::testing::Values(TestSizeVec {2, 4, 3, 2, 2, 2})));

/**
 * Test Parameters:
 * ----------------
 * (0) syn data type
 * (1) sizes before expand
 * (2) sizes after expand
 * (3) expand params (expand axis)
 */
class SynGaudi64BitExpandDimsTests
: public SynGaudi64BitMemoryOpTests
, public ::testing::WithParamInterface<std::tuple<synDataType, std::tuple<TestSizeVec, TestSizeVec, unsigned>>>
{
public:
protected:
};

TEST_P_GC(SynGaudi64BitExpandDimsTests, 64b_expand_dims_L2)
{
    const auto&         dtype            = std::get<0>(GetParam());
    const auto&         params           = std::get<1>(GetParam());
    const auto&         sizesBefore      = std::get<0>(params);
    const auto&         sizesAfter       = std::get<1>(params);
    const auto&         expandDimsAxis   = std::get<2>(params);
    synExpandDimsParams expandDimsParams = {.axis = expandDimsAxis};

    EXPECT_TRUE((dtype & mask64bDataType) != 0) << "Input type isn't 64b";
    EXPECT_TRUE(expandDimsParams.axis < sizesAfter.size()) << "Invalid expand dims axis param";

    auto expandDimsIn = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            const_cast<unsigned*>(sizesBefore.data()),
                                            sizesBefore.size(),
                                            dtype,
                                            nullptr,
                                            "expand_dims_in");

    auto expandDimsOut = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             const_cast<unsigned*>(sizesAfter.data()),
                                             sizesAfter.size(),
                                             dtype,
                                             nullptr,
                                             "expand_dims_out");

    addNodeToGraph(NodeFactory::expandDimsNodeTypeName,
                   {expandDimsIn},
                   {expandDimsOut},
                   &expandDimsParams,
                   sizeof(expandDimsParams),
                   "expand_dims");

    compileAndRun();

    const auto nElem = getNumberOfElements(sizesBefore.data(), sizesBefore.size());
    validateMemOp(expandDimsIn, expandDimsOut, nElem, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    low_rank,
    SynGaudi64BitExpandDimsTests,
    ::testing::Combine(::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
                       ::testing::Values(make_tuple(TestSizeVec {5}, TestSizeVec {1, 5}, 0),
                                         make_tuple(TestSizeVec {5}, TestSizeVec {5, 1}, 1),
                                         make_tuple(TestSizeVec {2, 4, 3}, TestSizeVec {1, 2, 4, 3}, 0),
                                         make_tuple(TestSizeVec {2, 4, 3}, TestSizeVec {2, 1, 4, 3}, 1),
                                         make_tuple(TestSizeVec {2, 4, 3}, TestSizeVec {2, 4, 1, 3}, 2),
                                         make_tuple(TestSizeVec {2, 4, 3}, TestSizeVec {2, 4, 3, 1}, 3))));

INSTANTIATE_TEST_SUITE_P(
    high_rank,
    SynGaudi64BitExpandDimsTests,
    ::testing::Combine(
        ::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
        ::testing::Values(make_tuple(TestSizeVec {2, 3, 4, 5, 6}, TestSizeVec {2, 3, 4, 5, 6, 1}, 5),
                          make_tuple(TestSizeVec {2, 3, 4, 5, 6, 5}, TestSizeVec {2, 3, 4, 5, 6, 5, 1}, 6))));

/**
 * Test Parameters:
 * ----------------
 * (0) syn data type
 * (1) sizes before expand
 * (2) sizes after expand
 * (3) squeeze params (squeeze axis)
 */
class SynGaudi64BitSqueezeTests
: public SynGaudi64BitMemoryOpTests
, public ::testing::WithParamInterface<std::tuple<synDataType, std::tuple<TestSizeVec, TestSizeVec, unsigned>>>
{
public:
protected:
};

TEST_P_GC(SynGaudi64BitSqueezeTests, 64b_squeeze)
{
    const auto&         dtype         = std::get<0>(GetParam());
    const auto&         params        = std::get<1>(GetParam());
    const auto&         sizesBefore   = std::get<0>(params);
    const auto&         sizesAfter    = std::get<1>(params);
    const auto&         squeezeAxis   = std::get<2>(params);
    synExpandDimsParams squeezeParams = {.axis = squeezeAxis};

    EXPECT_TRUE((dtype & mask64bDataType) != 0) << "Input type isn't 64b";
    EXPECT_TRUE(squeezeParams.axis < sizesBefore.size()) << "Invalid squeeze axis param";

    auto squeezeIn = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         const_cast<unsigned*>(sizesBefore.data()),
                                         sizesBefore.size(),
                                         dtype,
                                         nullptr,
                                         "squeeze_in");

    auto squeezeOut = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          const_cast<unsigned*>(sizesAfter.data()),
                                          sizesAfter.size(),
                                          dtype,
                                          nullptr,
                                          "squeeze_out");

    addNodeToGraph(NodeFactory::squeezeNodeTypeName,
                   {squeezeIn},
                   {squeezeOut},
                   &squeezeParams,
                   sizeof(squeezeParams),
                   "squeeze");

    compileAndRun();

    const auto nElem = getNumberOfElements(sizesBefore.data(), sizesBefore.size());
    validateMemOp(squeezeIn, squeezeOut, nElem, dtype);
}

INSTANTIATE_TEST_SUITE_P(
    low_rank,
    SynGaudi64BitSqueezeTests,
    ::testing::Combine(::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
                       ::testing::Values(make_tuple(TestSizeVec {1, 5}, TestSizeVec {5}, 0),
                                         make_tuple(TestSizeVec {5, 1}, TestSizeVec {5}, 1),
                                         make_tuple(TestSizeVec {1, 2, 4, 3}, TestSizeVec {2, 4, 3}, 0),
                                         make_tuple(TestSizeVec {2, 1, 4, 3}, TestSizeVec {2, 4, 3}, 1),
                                         make_tuple(TestSizeVec {2, 4, 1, 3}, TestSizeVec {2, 4, 3}, 2),
                                         make_tuple(TestSizeVec {2, 4, 3, 1}, TestSizeVec {2, 4, 3}, 3))));

INSTANTIATE_TEST_SUITE_P(
    high_rank,
    SynGaudi64BitSqueezeTests,
    ::testing::Combine(
        ::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
        ::testing::Values(make_tuple(TestSizeVec {2, 3, 4, 5, 6, 1}, TestSizeVec {2, 3, 4, 5, 6}, 5),
                          make_tuple(TestSizeVec {2, 3, 4, 5, 6, 5, 1}, TestSizeVec {2, 3, 4, 5, 6, 5}, 6))));

/**
 * Test Parameters:
 * ----------------
 * (0) syn data type
 * (1) sizes before reshape
 * (2) sizes after reshape
 */
class SynGaudi64BitReshapeTests
: public SynGaudi64BitMemoryOpTests
, public ::testing::WithParamInterface<std::tuple<synDataType, std::tuple<TestSizeVec, TestSizeVec>>>
{
public:
protected:
};

TEST_P_GC(SynGaudi64BitReshapeTests, 64b_reshape)
{
    const auto& dtype       = std::get<0>(GetParam());
    const auto& params      = std::get<1>(GetParam());
    const auto& sizesBefore = std::get<0>(params);
    const auto& sizesAfter  = std::get<1>(params);

    EXPECT_TRUE((dtype & mask64bDataType) != 0) << "Input type isn't 64b";

    auto reshapeIn = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         const_cast<unsigned*>(sizesBefore.data()),
                                         sizesBefore.size(),
                                         dtype,
                                         nullptr,
                                         "reshape_in");

    auto reshapeOut = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          const_cast<unsigned*>(sizesAfter.data()),
                                          sizesAfter.size(),
                                          dtype,
                                          nullptr,
                                          "reshape_out");

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {reshapeIn}, {reshapeOut}, nullptr, 0, "reshape");

    compileAndRun();

    const auto nElem = getNumberOfElements(sizesBefore.data(), sizesBefore.size());
    validateMemOp(reshapeIn, reshapeOut, nElem, dtype);
}

INSTANTIATE_TEST_SUITE_P(low_rank,
                         SynGaudi64BitReshapeTests,
                         ::testing::Combine(::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
                                            ::testing::Values(make_tuple(TestSizeVec {8}, TestSizeVec {2, 2, 2}),
                                                              make_tuple(TestSizeVec {4, 7, 8}, TestSizeVec {2, 7, 16}))));

INSTANTIATE_TEST_SUITE_P(
    high_rank,
    SynGaudi64BitReshapeTests,
    ::testing::Combine(::testing::ValuesIn({syn_type_int64, syn_type_uint64}),
                       ::testing::Values(make_tuple(TestSizeVec {4, 2, 2, 2, 2}, TestSizeVec {2, 2, 2, 2, 2, 2}),
                                         make_tuple(TestSizeVec {2, 2, 2, 2, 2, 2}, TestSizeVec {4, 2, 2, 2, 2}))));

class SynGaudiIdentityTestCasting : public SynGaudiTestInfra
{
};

TEST_F_GC(SynGaudiIdentityTestCasting, identity64u_64i)
{
    // TODO: Remove once [SW-136998] is done
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    const TestSizeVec sizes = {3, 2};

    const auto inDatatype  = syn_type_uint64;
    const auto outDatatype = syn_type_int64;
    const auto dim         = sizes.size();

    const auto tensorIn = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              const_cast<unsigned*>(sizes.data()),
                                              dim,
                                              inDatatype,
                                              nullptr,
                                              "identity_in");

    const auto tensorOut = createPersistTensor(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               const_cast<unsigned*>(sizes.data()),
                                               dim,
                                               outDatatype,
                                               nullptr,
                                               "identity_out");

    addNodeToGraph(NodeFactory::identityNodeTypeName, {tensorIn}, {tensorOut}, nullptr, 0, "identity64u_64i");
    compileAndRun();
}

class SynGaudi64BitMemsetTests : public SynGaudiTestInfra
{
};

TEST_F_GC(SynGaudi64BitMemsetTests, memset_dense, {synDeviceGaudi, synDeviceGaudi2})
{
    unsigned sizes[]  = {128, 24, 2};
    unsigned elements = std::accumulate(std::begin(sizes), std::end(sizes), (unsigned)1, std::multiplies<>());

    unsigned out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes, 3, syn_type_int64);

    addNodeToGraph(NodeFactory::memsetNodeTypeName, {}, {out}, nullptr, 0);

    compileTopology();
    runTopology(0, true);

    auto* outPtr = castHostBuffer<int64_t>(out);
    for (unsigned idx = 0; idx < elements; ++idx)
    {
        ASSERT_EQ(outPtr[idx], 0);
    }
}

TEST_F_GC(SynGaudi64BitMemsetTests, memset_sparse, {synDeviceGaudi2})
{
    GlobalConfTestSetter gConvVar("ENABLE_INTERNAL_NODES", "true");

    unsigned zerosSizes[] = {2, 128, 4, 2};
    unsigned onesSizes[]  = {1, 128, 4, 2};
    unsigned outSizes[]   = {3, 128, 4, 2};
    unsigned elements     = std::accumulate(std::begin(outSizes), std::end(outSizes), (unsigned)1, std::multiplies<>());

    unsigned zeros = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, zerosSizes, 4, syn_type_int64);
    unsigned ones  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, onesSizes, 4, syn_type_int64);
    unsigned out   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, outSizes, 4, syn_type_int64);

    addNodeToGraph(NodeFactory::memsetNodeTypeName, TensorIndices({}), TensorIndices({zeros}), nullptr, 0);

    synAxisParams p = {.axis = 0};
    addNodeToGraph(NodeFactory::concatenateNodeLogicalInternalTypeName, {zeros, ones}, {out}, (void*)&p, sizeof(p));

    compileTopology();
    runTopology(0, true);

    auto* outPtr = castHostBuffer<int64_t>(out);
    for (unsigned idx = 0; idx < elements; ++idx)
    {
        ASSERT_EQ(outPtr[idx], (idx % 3 == 2) ? 1 : 0) << "mismatch at index: " << idx;
    }
}
