#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "syn_data_type_type_conversions.h"
#include "types.h"
#include "gtest/gtest.h"


class SynGaudiDynamicReshapeTest
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<::testing::tuple<unsigned,   // 0
                                                      unsigned,   // 1
                                                      unsigned,   // 2
                                                      unsigned,   // 3
                                                      unsigned,   // 4
                                                      unsigned,   // 5
                                                      unsigned,   // 6
                                                      unsigned>>  // 7
{
public:
    struct GetName
    {
        template <class ParamType>
        std::string operator()(const ::testing::TestParamInfo<ParamType> &info) const
        {
            ::std::stringstream ss;
            ss << "ifm_DxWxHxB_ofm_DxWxHxB_"
               << std::get<0>(info.param) << "x"
               << std::get<1>(info.param) << "x"
               << std::get<2>(info.param) << "x"
               << std::get<3>(info.param) << "_"
               << std::get<4>(info.param) << "x"
               << std::get<5>(info.param) << "x"
               << std::get<6>(info.param) << "x"
               << std::get<7>(info.param);
            return ss.str();
        }
    };

    template<typename DType=float>
    void reshapeTest()
    {
        ScopedConfigurationChange scc("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
        const unsigned            tensorDim = 4;

        TestSizes inMaxSize     = {1, 1, 1, 1, 1};
        TestSizes inMinSize     = {1, 1, 1, 1, 1};
        TestSizes actualInSize  = {1, 1, 1, 1, 1};
        TestSizes outMaxSize    = {1, 1, 1, 1, 1};
        TestSizes outMinSize    = {1, 1, 1, 1, 1};
        TestSizes actualOutSize = {1, 1, 1, 1, 1};

        inMaxSize[0] = ::testing::get<0>(GetParam()) * 2;
        inMaxSize[1] = ::testing::get<1>(GetParam()) * 2;
        inMaxSize[2] = ::testing::get<2>(GetParam()) * 2;
        inMaxSize[3] = ::testing::get<3>(GetParam()) * 2;

        inMinSize[0] = ::testing::get<0>(GetParam());
        inMinSize[1] = ::testing::get<1>(GetParam());
        inMinSize[2] = ::testing::get<2>(GetParam());
        inMinSize[3] = ::testing::get<3>(GetParam());

        actualInSize[0] = ::testing::get<0>(GetParam());
        actualInSize[1] = ::testing::get<1>(GetParam());
        actualInSize[2] = ::testing::get<2>(GetParam());
        actualInSize[3] = ::testing::get<3>(GetParam());


        outMaxSize[0] = ::testing::get<4>(GetParam()) * 2;
        outMaxSize[1] = ::testing::get<5>(GetParam()) * 2;
        outMaxSize[2] = ::testing::get<6>(GetParam()) * 2;
        outMaxSize[3] = ::testing::get<7>(GetParam()) * 2;

        outMinSize[0] = ::testing::get<4>(GetParam());
        outMinSize[1] = ::testing::get<5>(GetParam());
        outMinSize[2] = ::testing::get<6>(GetParam());
        outMinSize[3] = ::testing::get<7>(GetParam());

        actualOutSize[0] = ::testing::get<4>(GetParam());
        actualOutSize[1] = ::testing::get<5>(GetParam());
        actualOutSize[2] = ::testing::get<6>(GetParam());
        actualOutSize[3] = ::testing::get<7>(GetParam());

        unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                inMaxSize.data(),
                                                tensorDim,
                                                asSynType<DType>(),
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                inMinSize.data());

        unsigned outTensorJustForSizes =
                createShapeTensor(INPUT_TENSOR, outMaxSize.data(), outMinSize.data(), tensorDim, syn_type_single);

        unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outMaxSize.data(),
                                                 tensorDim,
                                                 asSynType<DType>(),
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outMinSize.data());

        addNodeToGraph("reshape", {inTensor, outTensorJustForSizes}, {outTensor});

        compileTopology();
        setActualSizes(inTensor, actualInSize.data());
        setActualSizes(outTensor, actualOutSize.data());
        setActualSizes(outTensorJustForSizes, actualOutSize.data());
        runTopology(0, true);

        auto* inBuffer  = castHostInBuffer<DType>(inTensor);
        auto* outBuffer = castHostOutBuffer<DType>(outTensor);
        // Test by the actual batch size.
        std::queue<DType> data  = serializeBuffer(inBuffer, inMaxSize, actualInSize);
        std::queue<DType> data2 = serializeBuffer(outBuffer, outMaxSize, actualOutSize);
        auto              size  = data.size();
        for (int i = 0; i < size; i++)
        {
            ASSERT_EQ(data.front(), data2.front()) << i;
            data.pop();
            data2.pop();
        }
    }
};

TEST_P_GC(SynGaudiDynamicReshapeTest, tpc_reshape)
{
    reshapeTest();
}

TEST_P_GC(SynGaudiDynamicReshapeTest, dma_reshape, {synDeviceGaudi, synDeviceGaudi2})
{
    ScopedConfigurationChange use_dma("USE_DMA_IN_PHYSICAL_RESHAPE", "true");
    reshapeTest();
}

TEST_P_GC(SynGaudiDynamicReshapeTest, dma_reshape_int64, {synDeviceGaudi, synDeviceGaudi2})
{
    ScopedConfigurationChange use_dma("USE_DMA_IN_PHYSICAL_RESHAPE", "true");
    reshapeTest<int64_t>();
}

TEST_P_GC(SynGaudiDynamicReshapeTest, dma_reshape_uint64, {synDeviceGaudi, synDeviceGaudi2})
{
    ScopedConfigurationChange use_dma("USE_DMA_IN_PHYSICAL_RESHAPE", "true");
    reshapeTest<uint64_t>();
}

TEST_P_GC(SynGaudiDynamicReshapeTest, tpc_reshape_bf16)
{
    reshapeTest<bfloat16>();
}

TEST_P_GC(SynGaudiDynamicReshapeTest, dma_reshape_bf16, {synDeviceGaudi, synDeviceGaudi2})
{
    ScopedConfigurationChange use_dma("USE_DMA_IN_PHYSICAL_RESHAPE", "true");
    reshapeTest<bfloat16>();
}


INSTANTIATE_TEST_SUITE_P(,
    SynGaudiDynamicReshapeTest,
    ::testing::Values(
        ::testing::make_tuple(66, 3, 1, 1, 18, 1, 11, 1),
        ::testing::make_tuple(67, 1, 4, 1, 134, 2, 1, 1),
        ::testing::make_tuple(2007, 1, 28, 1, 14049,  4, 1, 1),
        ::testing::make_tuple(14049, 1, 1, 1, 2007, 7, 1, 1),
        ::testing::make_tuple(2024, 1, 96, 1, 48576, 4, 1, 1),
        ::testing::make_tuple(48576, 1, 1, 1, 2024, 24, 1, 1)
    ),
    SynGaudiDynamicReshapeTest::GetName()
);

class SynGaudiDynamicLogicalReshapeTest
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<::testing::tuple<TestSizes, TestSizes, TestSizes,TestSizes, TestSizes, TestSizes >>
{
public:
    struct GetName
    {
        template <class ParamType>
        std::string operator()(const ::testing::TestParamInfo<ParamType> &info) const
        {
            ::std::stringstream ss;
            ss << "inmax_inmin_inactual_outmax_outmin_outactual_"
               << toString(::testing::get<0>(info.param), 'x') << "_"
               << toString(::testing::get<1>(info.param), 'x') << "_"
               << toString(::testing::get<2>(info.param), 'x') << "_"
               << toString(::testing::get<3>(info.param), 'x') << "_"
               << toString(::testing::get<4>(info.param), 'x') << "_"
               << toString(::testing::get<5>(info.param), 'x');
            return ss.str();
        }
    };
};

TEST_P_GC(SynGaudiDynamicLogicalReshapeTest, reshape)
{
    ScopedConfigurationChange scc("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    const unsigned            tensorDim = 4;

    TestSizes inMaxSize     = ::testing::get<0>(GetParam());
    TestSizes inMinSize     = ::testing::get<1>(GetParam());
    TestSizes actualInSize  = ::testing::get<2>(GetParam());
    TestSizes outMaxSize    = ::testing::get<3>(GetParam());
    TestSizes outMinSize    = ::testing::get<4>(GetParam());
    TestSizes actualOutSize = ::testing::get<5>(GetParam());

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSize.data(),
                                            tensorDim,
                                            syn_type_single,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSize.data());

    unsigned outTensorJustForSizes =
        createShapeTensor(INPUT_TENSOR, outMaxSize.data(), outMinSize.data(), tensorDim, syn_type_single);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             outMaxSize.data(),
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSize.data());

    addNodeToGraph("reshape", {inTensor, outTensorJustForSizes}, {outTensor});

    compileTopology();
    setActualSizes(inTensor, actualInSize.data());
    setActualSizes(outTensor, actualOutSize.data());
    setActualSizes(outTensorJustForSizes, actualOutSize.data());
    runTopology(0, true);

    float* inBuffer  = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);
    // Test by the actual batch size.
    std::queue<float> data  = serializeBuffer(inBuffer, inMaxSize, actualInSize);
    std::queue<float> data2 = serializeBuffer(outBuffer, outMaxSize, actualOutSize);
    auto              size  = data.size();
    for (int i = 0; i < size; i++)
    {
        ASSERT_EQ(data.front(), data2.front()) << i;
        data.pop();
        data2.pop();
    }
}


INSTANTIATE_TEST_SUITE_P(,
    SynGaudiDynamicLogicalReshapeTest,
    ::testing::Values(
        ::testing::make_tuple(
            TestSizes{5, 5, 5, 5, 1},
            TestSizes{5, 5, 2, 5, 1},
            TestSizes{5, 5, 3, 5, 1},
            TestSizes{25, 5, 5, 1, 1},
            TestSizes{25, 2, 5, 1, 1},
            TestSizes{25, 3, 5, 1, 1}
        ),
        ::testing::make_tuple(
            TestSizes{6, 5, 7, 11, 1},
            TestSizes{6, 5, 2, 11, 1},
            TestSizes{6, 5, 3, 11, 1},
            TestSizes{2, 15, 7, 11, 1},
            TestSizes{2, 15, 2, 11, 1},
            TestSizes{2, 15, 3, 11, 1}
        )
    ),
    SynGaudiDynamicLogicalReshapeTest::GetName()
);
