#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "gc_gaudi_test_infra.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "synapse_common_types.h"

class SynGaudiReinterpretCastTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<unsigned>  // nofDims
{
};

INSTANTIATE_TEST_SUITE_P(reinterpretcast, SynGaudiReinterpretCastTest, testing::Range(2u, 9u));

TEST_P_GC(SynGaudiReinterpretCastTest, reinterpret_cast_static_test, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    constexpr unsigned    size = (1 << 10);
    std::vector<unsigned> inSizes(GetParam(), 2);
    std::vector<unsigned> outSizes(GetParam(), 2);
    std::vector<unsigned> middSizes(GetParam(), 2);

    inSizes[0]   = 32;
    inSizes[1]   = size;
    outSizes[0]  = 2;
    outSizes[1]  = size;
    middSizes[0] = 1;
    middSizes[1] = size;

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizes.data(),
                                      inSizes.size(),
                                      syn_type_bf16);
    unsigned slice =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), syn_type_bf16);
    unsigned midd1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, middSizes.data(), middSizes.size());
    unsigned midd2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, middSizes.data(), middSizes.size());
    unsigned ref =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), syn_type_bf16);
    unsigned out =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes.data(), outSizes.size(), syn_type_bf16);

    synSliceParamsNDims p;
    p.axes[0] = 0;
    p.axes[1] = 1;

    p.steps[0] = 1;
    p.steps[1] = 1;

    p.starts[0] = 0;
    p.starts[1] = 0;

    p.ends[0] = 2;
    p.ends[1] = size;

    for (unsigned d = 2; d < GetParam(); d++)
    {
        p.axes[d]   = d;
        p.steps[d]  = 1;
        p.starts[d] = 0;
        p.ends[d]   = 2;
    }

    addNodeToGraph("slice", {in}, {slice}, (void*)&p, sizeof(p));

    // sequence A
    addNodeToGraph("memcpy", {slice}, {ref});
    // sequence B
    addNodeToGraph(NodeFactory::reinterpretCastNodeTypeName, {slice}, {midd1});
    addNodeToGraph("memcpy", {midd1}, {midd2});
    addNodeToGraph(NodeFactory::reinterpretCastNodeTypeName, {midd2}, {out});

    compileAndRun();

    char* refBuffer = (char*)m_hostBuffers[ref];
    char* outBuffer = (char*)m_hostBuffers[out];

    uint64_t nofElements = multiplyElements(outSizes);
    for (uint64_t idx = 0; idx < 2 * nofElements; ++idx)
    {
        ASSERT_EQ(refBuffer[idx], outBuffer[idx]) << "OUTPUT: Mismatch for at index " << idx / 2
                                                  << " |Expected: " << refBuffer[idx] << " |Result: " << outBuffer[idx];
    }
}

// DSD and NDim are not supported together, so not checking NDim for the dynamic case
TEST_F_GC(SynGaudiReinterpretCastTest, reinterpret_cast_dynamic_test, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    constexpr unsigned size             = (1 << 20);
    unsigned           inMaxSizes[]     = {32, size};
    unsigned           inMinSizes[]     = {32, size >> 2};
    unsigned           inActualSizes[]  = {32, size >> 1};
    unsigned           outMaxSizes[]    = {8, size};
    unsigned           outMinSizes[]    = {2, size >> 2};
    unsigned           outActualSizes[] = {6, size >> 1};
    unsigned           middMaxSizes[]   = {4, size};
    unsigned           middMinSizes[]   = {1, size >> 2};

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
    unsigned shape = createShapeTensor(OUTPUT_TENSOR, outMaxSizes, outMinSizes, ARRAY_SIZE(outMaxSizes));
    unsigned slice = createTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ZERO,
                                  nullptr,
                                  outMaxSizes,
                                  ARRAY_SIZE(outMaxSizes),
                                  syn_type_bf16,
                                  nullptr,
                                  outMinSizes);
    unsigned midd1 = createTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ZERO,
                                  nullptr,
                                  middMaxSizes,
                                  ARRAY_SIZE(middMaxSizes),
                                  syn_type_float,
                                  nullptr,
                                  middMinSizes);
    unsigned midd2 = createTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ZERO,
                                  nullptr,
                                  middMaxSizes,
                                  ARRAY_SIZE(middMaxSizes),
                                  syn_type_float,
                                  nullptr,
                                  middMinSizes);
    unsigned ref = createPersistTensor(OUTPUT_TENSOR,
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
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
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

    p.ends[0] = 8;
    p.ends[1] = size;

    addNodeToGraph("slice", {in, shape}, {slice}, (void*)&p, sizeof(p));

    // sequence A
    addNodeToGraph("memcpy", {slice}, {ref});
    // sequence B
    addNodeToGraph(NodeFactory::reinterpretCastNodeTypeName, {slice}, {midd1});
    addNodeToGraph("memcpy", {midd1}, {midd2});
    addNodeToGraph(NodeFactory::reinterpretCastNodeTypeName, {midd2}, {out});

    compileTopology();

    setActualSizes(in, inActualSizes);
    setActualSizes(shape, outActualSizes);
    setActualSizes(ref, outActualSizes);
    setActualSizes(out, outActualSizes);

    runTopology();

    char* refBuffer = (char*)m_hostBuffers[ref];
    char* outBuffer = (char*)m_hostBuffers[out];

    for (uint64_t idx = 0; idx < (uint64_t)2 * 2 * size; ++idx)
    {
        ASSERT_EQ(refBuffer[idx], outBuffer[idx]) << "OUTPUT: Mismatch for at index " << idx / 2
                                                  << " |Expected: " << refBuffer[idx] << " |Result: " << outBuffer[idx];
    }
}
