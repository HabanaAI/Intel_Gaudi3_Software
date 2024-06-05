#include <stdint.h>
#include <memory>
#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "scoped_configuration_change.h"
#include "node_factory.h"
#include "dma_node_test.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "synapse_common_types.h"

TEST_F_GC(SynGaudiTestDma, linear_dma_node)
{
    linear_dma_node();
}

// Linear dma between two other (relu) nodes
TEST_F_GC(SynGaudiTestDma, relu_forward_and_backward_with_linear_dma)
{
    relu_forward_and_backward_with_linear_dma();
}

TEST_F_GC(SynGaudiTestDma, strided_dma_node)
{
    strided_dma_node();
}

TEST_F_GC(SynGaudiTestDma, three_dimensional_strided_dma_node_L2)
{
    three_dimensional_strided_dma_node();
}

TEST_F_GC(SynGaudiTestDma, three_dimensional_strided_dma_node)
{
    three_dimensional_strided_dma_node();
}

TEST_F_GC(SynGaudiTestDmaMemset, linear_memset_dma_node)
{
    linear_memset_dma_node();
}

TEST_F_GC(SynGaudiTestDmaMemset, linear_memset_2d_dma_node)
{
    linear_memset_2d_dma_node();
}

TEST_F_GC(SynGaudiTestDmaMemset, three_dimensional_memset_node)
{
    three_dimensional_memset_node();
}

TEST_P_GC(SynGaudiTestDmaMemset, dma_memset_full_coverage_DAILY)
{
    linear_memset_dma_node_single_test();
}

INSTANTIATE_TEST_SUITE_P(dma_memset_small_size,
                        SynGaudiTestDmaMemset,
                        ::testing::Combine(
                            ::testing::Range(1,257),    // size
                            ::testing::Range(1,5)));  // parallel level


INSTANTIATE_TEST_SUITE_P(dma_memset_unaligned,
                        SynGaudiTestDmaMemset,
                        ::testing::Combine(
                            ::testing::Range(1000,34*1024,1024),    //size
                            ::testing::Range(1,5)));                // parallel level;

INSTANTIATE_TEST_SUITE_P(dma_memset_large_aligned,
                        SynGaudiTestDmaMemset,
                        ::testing::Combine(
                        ::testing::Range(30*1024,0x100000,10*1024),    // size
                        ::testing::Range(1,5)));                      // parallel level;

TEST_F_GC(SynGaudiTestInfra, long_graph_with_tpc_and_dma, {synDeviceGaudi2})
{
    ScopedConfigurationChange removeMemcpy("ENABLE_REMOVE_REDUNDANT_MEMCPY", "false");

    static const unsigned numNodes = 100;

    unsigned reluIn        = 0;
    unsigned dmaOut        = 0;
    unsigned rollingIdxIn  = 0;
    unsigned rollingIdxOut = 0;

    // Graph will have this structure: [relu_fwd]->[memcpy]->[relu_fwd]->[memcpy]->[relu_fwd]-> ... ->[memcpy]
    reluIn        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE);
    rollingIdxOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);
    addNodeToGraph("relu_fwd_f32", {reluIn}, {rollingIdxOut});
    for (unsigned i = 0; i < numNodes; i++)
    {
        rollingIdxIn  = connectOutputTensorToInputTensor(rollingIdxOut);
        rollingIdxOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);
        addNodeToGraph("memcpy", {rollingIdxIn}, {rollingIdxOut});
        rollingIdxIn  = connectOutputTensorToInputTensor(rollingIdxOut);
        rollingIdxOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);
        addNodeToGraph("relu_fwd_f32", {rollingIdxIn}, {rollingIdxOut});
    }
    rollingIdxIn = connectOutputTensorToInputTensor(rollingIdxOut);
    dmaOut       = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO);
    addNodeToGraph("memcpy", {rollingIdxIn}, {dmaOut});

    compileTopology();
    runTopology();

    auto* input  = m_hostBuffers[reluIn];
    auto* output = m_hostBuffers[dmaOut];

    ASSERT_EQ(memcmp(input, output, getDefaultNumberOfElements() * sizeof(float)), 0);
}

class ArcSyncSchemeResetProcedure
: public SynGaudiTestInfra
, public testing::WithParamInterface<int>
{
};

TEST_P_GC(ArcSyncSchemeResetProcedure, signal_limit_is_parameterized, {synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange arcSupportMode("ARC_SYNC_SCHEME_SIGNAL_LIMIT", std::to_string(GetParam()));
    ScopedConfigurationChange removeMemcpy("ENABLE_REMOVE_REDUNDANT_MEMCPY", "false");

    static const unsigned numInternalPairNodes = 10;  // 22 nodes in total in the graph

    const unsigned dims          = 1;
    unsigned       size[dims]    = {1024 * 32 * 10};
    unsigned       memcpyIn      = 0;
    unsigned       reluOut       = 0;
    unsigned       rollingIdxIn  = 0;
    unsigned       rollingIdxOut = 0;

    // Graph will have this structure: [memcpy]->[relu_fwd]->[memcpy]->[relu_fwd]-> ... ->[relu_fwd]
    memcpyIn      = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, size, dims);
    rollingIdxOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, size, dims);
    addNodeToGraph("memcpy", {memcpyIn}, {rollingIdxOut});
    for (unsigned i = 0; i < numInternalPairNodes; i++)
    {
        rollingIdxIn  = connectOutputTensorToInputTensor(rollingIdxOut);
        rollingIdxOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, size, dims);
        addNodeToGraph("relu_fwd_f32", {rollingIdxIn}, {rollingIdxOut});
        rollingIdxIn  = connectOutputTensorToInputTensor(rollingIdxOut);
        rollingIdxOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, size, dims);
        addNodeToGraph("memcpy", {rollingIdxIn}, {rollingIdxOut});
    }
    rollingIdxIn = connectOutputTensorToInputTensor(rollingIdxOut);
    reluOut      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, size, dims);
    addNodeToGraph("relu_fwd_f32", {rollingIdxIn}, {reluOut});

    compileTopology();
    runTopology();

    auto* input  = m_hostBuffers[memcpyIn];
    auto* output = m_hostBuffers[reluOut];

    ASSERT_EQ(memcmp(input, output, getDefaultNumberOfElements() * sizeof(float)), 0);
}
INSTANTIATE_TEST_SUITE_P(, ArcSyncSchemeResetProcedure, ::testing::Range(5, 28));

class SynGaudiEagerMemcpy : public SynGaudiTestDma
{
public:
    SynGaudiEagerMemcpy()
    {
        m_testConfig.m_compilationMode = COMP_EAGER_MODE_TEST;
        m_testConfig.m_supportedDeviceTypes.clear();
        setSupportedDevices({synDeviceGaudi2});
        setTestPackage(TEST_PACKAGE_EAGER);
    }
};

TEST_F_GC(SynGaudiEagerMemcpy, single_memcpy_test)
{
    linear_dma_node();
}

class SynGaudiEagerMemset : public SynGaudiTestDmaMemset
{
public:
    SynGaudiEagerMemset()
    {
        m_testConfig.m_compilationMode = COMP_EAGER_MODE_TEST;
        m_testConfig.m_supportedDeviceTypes.clear();
        setSupportedDevices({synDeviceGaudi2});
        setTestPackage(TEST_PACKAGE_EAGER);
    }
};

TEST_F_GC(SynGaudiEagerMemset, single_memset_test)
{
    linear_memset_dma_node();
}

class dynamicStridedMemcpyOnTransposeEngineTest : public SynGaudiTwoRunCompareTest
{
};

TEST_F_GC(dynamicStridedMemcpyOnTransposeEngineTest, dynamic_strided_memcpy_on_transpose_engine, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned inMaxSizes[]     = {105, 1150};
    unsigned inMinSizes[]     = {105, 115};
    unsigned inActualSizes[]  = {105, 500};
    unsigned outMaxSizes[]    = {1, 1150};
    unsigned outMinSizes[]    = {1, 115};
    unsigned outActualSizes[] = {1, 500};

    unsigned in    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inMaxSizes,
                                      2,
                                      syn_type_bf16,
                                      nullptr,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      inMinSizes);
    unsigned shape = createShapeTensor(INPUT_TENSOR, outMaxSizes, outMinSizes, 2);
    unsigned out   = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outMaxSizes,
                                       2,
                                       syn_type_bf16,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       outMinSizes);

    synSliceParams p;
    p.axes[0] = 0;
    p.axes[1] = 1;

    p.steps[0] = 1;
    p.steps[1] = 1;

    p.starts[0] = 0;
    p.starts[1] = 0;

    p.ends[0] = 1;
    p.ends[1] = 1150;

    addNodeToGraph("slice", {in, shape}, {out}, (void*)&p, sizeof(p));

    addConfigurationToRun(FIRST_RUN, "ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE", "true");

    setActualSizes(in, inActualSizes);
    setActualSizes(out, outActualSizes);
    setActualSizes(shape, outActualSizes);

    compareRunsResults({out});
}
