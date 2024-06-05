#include "gc_gaudi_test_infra.h"
#include "graph_compiler/sim_graph.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "data_type_utils.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "transpose_utils.h"
#include "syn_gaudi_two_run_compare_test.h"

class SynInferMaxShapeTest : public SynGaudiTestInfra
{
};

TEST_F_GC(SynInferMaxShapeTest, transpose_test)
{
    GlobalConfTestSetter s2("ENABLE_INTERNAL_NODES", "true");

    unsigned inMaxSizes[]  = {33, 65, 129};
    unsigned inMinSizes[]  = {17, 33, 65};
    unsigned outMaxSizes[] = {129, 65, 33};
    unsigned outMinSizes[] = {65, 33, 17};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inMaxSizes,
                                      3,
                                      syn_type_float,
                                      nullptr,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      inMinSizes);

    unsigned inShape  = createShapeTensor(OUTPUT_TENSOR, inMaxSizes, inMinSizes, 3);
    unsigned outShape = createShapeTensor(OUTPUT_TENSOR, outMaxSizes, outMinSizes, 3);
    unsigned midd1    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inMaxSizes, 3, syn_type_float);
    unsigned midd2    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSizes, 3, syn_type_float);

    unsigned ref = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outMaxSizes,
                                       3,
                                       syn_type_float,
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
                                       3,
                                       syn_type_float,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       outMinSizes);

    TransposePermutationArray p;
    p.push_back((TransposePermutationDim)2);
    p.push_back((TransposePermutationDim)1);
    p.push_back((TransposePermutationDim)0);

    synTransposeParamsNDims tp  = permutationToParams(p);
    synTransposeParamsNDims tp2 = permutationToParams(p);
    synTransposeParamsNDims tp3 = permutationToParams(p);

    addNodeToGraph("infer_max_shape", {in}, {midd1, inShape});
    addNodeToGraph("transpose", {midd1}, {midd2}, (void*)&tp2, sizeof(tp2));
    addNodeToGraph("transpose_shape", {inShape}, {outShape}, (void*)&tp3, sizeof(tp3));
    addNodeToGraph("identity", {midd2, outShape}, {out});

    // ref
    addNodeToGraph("transpose", {in}, {ref}, (void*)&tp, sizeof(tp));

    compileTopology();

    setActualSizes(in, inMinSizes);
    setActualSizes(ref, outMinSizes);
    setActualSizes(out, outMinSizes);

    setAsInternalShapeTensor(inShape);
    setAsInternalShapeTensor(outShape);

    runTopology();

    float* pOutBuffer = (float*)m_hostBuffers[out];
    float* pRefBuffer = (float*)m_hostBuffers[ref];
    for (unsigned i = 0; i < 65 * 33 * 17; ++i)
    {
        ASSERT_FLOAT_EQ(pOutBuffer[i], pRefBuffer[i]) << "mismatch at index: " << i;
    }
}

TEST_F_GC(SynInferMaxShapeTest, neg_test)
{
    GlobalConfTestSetter s2("ENABLE_INTERNAL_NODES", "true");

    unsigned maxSizes[]    = {33, 65, 129};
    unsigned minSizes[]    = {17, 33, 65};
    unsigned actualSizes[] = {20, 40, 80};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      maxSizes,
                                      3,
                                      syn_type_float,
                                      nullptr,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      minSizes);

    unsigned shape = createShapeTensor(OUTPUT_TENSOR, maxSizes, minSizes, 3);
    unsigned midd1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 3, syn_type_float);
    unsigned midd2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 3, syn_type_float);
    unsigned midd3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 3, syn_type_float);
    unsigned midd4 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 3, syn_type_float);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       maxSizes,
                                       3,
                                       syn_type_float,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       minSizes);

    addNodeToGraph("infer_max_shape", {in}, {midd1, shape});
    addNodeToGraph("neg_fwd_f32", {midd1}, {midd2});
    addNodeToGraph("memcpy", {midd2}, {midd3});
    addNodeToGraph("neg_fwd_f32", {midd3}, {midd4});
    addNodeToGraph("identity", {midd4, shape}, {out});

    compileTopology();

    setActualSizes(in, actualSizes);
    setActualSizes(out, actualSizes);
    setAsInternalShapeTensor(shape);

    runTopology();

    float* pInBuffer  = (float*)m_hostBuffers[in];
    float* pOutBuffer = (float*)m_hostBuffers[out];
    for (unsigned i = 0; i < 20 * 40 * 80; ++i)
    {
        ASSERT_FLOAT_EQ(pOutBuffer[i], pInBuffer[i]) << "mismatch at index: " << i;
    }
}