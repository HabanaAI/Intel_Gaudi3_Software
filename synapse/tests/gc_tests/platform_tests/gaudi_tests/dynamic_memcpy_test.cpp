#include "syn_gaudi_two_run_compare_test.h"
#include "gc_dynamic_shapes_infra.h"
#include "gtest/gtest.h"

class SynGaudiDynamicMemcpyFcdAggrigationTest
: public SynGaudiTwoRunCompareTest
, public testing::WithParamInterface<
      std::tuple<std::tuple<std::vector<unsigned>, std::vector<unsigned>, std::vector<unsigned>>, synDataType>>
{
public:
    SynGaudiDynamicMemcpyFcdAggrigationTest();
    void runSingleTest();

protected:
    std::vector<unsigned> m_maxSizes;
    std::vector<unsigned> m_minSizes;
    std::vector<unsigned> m_actualSizes;
    synDataType           m_dataType;
};
SynGaudiDynamicMemcpyFcdAggrigationTest::SynGaudiDynamicMemcpyFcdAggrigationTest()
{
    const auto sizes = std::get<0>(GetParam());
    m_dataType       = std::get<1>(GetParam());
    m_maxSizes       = std::get<0>(sizes);
    m_minSizes       = std::get<1>(sizes);
    m_actualSizes    = std::get<2>(sizes);
}

void SynGaudiDynamicMemcpyFcdAggrigationTest::runSingleTest()
{
    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      m_maxSizes.data(),
                                      m_maxSizes.size(),
                                      m_dataType,
                                      nullptr,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      m_minSizes.data());

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_maxSizes.data(),
                                       m_maxSizes.size(),
                                       m_dataType,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       m_minSizes.data());

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {in}, {out});

    setActualSizes(in, m_actualSizes.data());
    setActualSizes(out, m_actualSizes.data());

    addConfigurationToRun(FIRST_RUN, "ENABLE_OPTIMIZE_MEMCPY_NODES", "true");
    addConfigurationToRun(FIRST_RUN, "ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_OPTIMIZE_MEMCPY_NODES", "false");
    addConfigurationToRun(SECOND_RUN, "GCFG_ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION", "false");

    compareRunsResults({out});
}

TEST_P_GC(SynGaudiDynamicMemcpyFcdAggrigationTest,
          different_aggrigated_dim_each_time,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(,
                         SynGaudiDynamicMemcpyFcdAggrigationTest,
                         ::testing::Combine(::testing::Values(std::make_tuple(std::vector<unsigned> {2, 2, 512, 128},
                                                                              std::vector<unsigned> {2, 2, 256, 128},
                                                                              std::vector<unsigned> {2, 2, 300, 128}),
                                                              std::make_tuple(std::vector<unsigned> {2, 2, 128, 512},
                                                                              std::vector<unsigned> {2, 2, 128, 256},
                                                                              std::vector<unsigned> {2, 2, 128, 300}),
                                                              std::make_tuple(std::vector<unsigned> {512, 128, 2, 2},
                                                                              std::vector<unsigned> {256, 128, 2, 2},
                                                                              std::vector<unsigned> {300, 128, 2, 2}),
                                                              std::make_tuple(std::vector<unsigned> {2, 512, 128, 2},
                                                                              std::vector<unsigned> {2, 256, 128, 2},
                                                                              std::vector<unsigned> {2, 300, 128, 2})),
                                            ::testing::ValuesIn({synDataType::syn_type_single,
                                                                 synDataType::syn_type_bf16})));

INSTANTIATE_TEST_SUITE_P(
    different_aggrigated_dim_each_time_ASIC_CI,
    SynGaudiDynamicMemcpyFcdAggrigationTest,
    ::testing::Combine(::testing::Values(std::make_tuple(std::vector<unsigned> {512, 512, 512, 4},
                                                         std::vector<unsigned> {512, 512, 256, 4},
                                                         std::vector<unsigned> {512, 512, 300, 4}),
                                         std::make_tuple(std::vector<unsigned> {512, 512, 4, 512},
                                                         std::vector<unsigned> {512, 512, 4, 256},
                                                         std::vector<unsigned> {512, 512, 4, 300}),
                                         std::make_tuple(std::vector<unsigned> {512, 4, 512, 512},
                                                         std::vector<unsigned> {256, 4, 512, 512},
                                                         std::vector<unsigned> {300, 4, 512, 512}),
                                         std::make_tuple(std::vector<unsigned> {512, 512, 4, 512},
                                                         std::vector<unsigned> {512, 256, 4, 512},
                                                         std::vector<unsigned> {512, 300, 4, 512})),
                       ::testing::ValuesIn({synDataType::syn_type_single, synDataType::syn_type_bf16})));

class SynGaudiTwoRunCompareTestWithREshapesGaudi3 : public SynGaudiTwoRunCompareTest
{
};

TEST_F_GC(SynGaudiTwoRunCompareTestWithREshapesGaudi3,
          dmaViaTransposeEng_wrap_with_reshapes_and_with_reinterpret_cast,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned maxSizes[]   = {2, 4, 70000};
    unsigned minSizes[]   = {2, 4, 68000};
    unsigned inStrides[]  = {2, 170, 680, 332180};
    unsigned outStrides[] = {2, 4, 16, 280000};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      maxSizes,
                                      3,
                                      syn_type_bf16,
                                      inStrides,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      minSizes);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       maxSizes,
                                       3,
                                       syn_type_bf16,
                                       outStrides,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       minSizes);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {in}, {out});

    unsigned actualSizes[3] = {2, 4, 69954};

    setActualSizes(in, actualSizes);
    setActualSizes(out, actualSizes);

    addConfigurationToRun(FIRST_RUN, "ENABLE_OPTIMIZE_MEMCPY_NODES", "true");
    addConfigurationToRun(FIRST_RUN, "ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_OPTIMIZE_MEMCPY_NODES", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE", "false");

    compareRunsResults({out});
}
