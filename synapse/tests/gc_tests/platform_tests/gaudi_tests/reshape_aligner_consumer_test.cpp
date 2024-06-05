#include "gaudi_tests/gc_dynamic_shapes_infra.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include "types.h"

class SynGaudiConsumerReshapeAlignTest : public SynGaudiTestInfra
{
};

TEST_F_GC(SynGaudiConsumerReshapeAlignTest,
          tpc_broadcasted_dyn_shape_consumer_reshape_align,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    TestSizes     gemmIn1MaxSizes = {1024, 1928};
    TestSizes     gemmIn1MinSizes = {1024, 964};
    TestSizes     gemmIn2MaxSizes = {1024, 1024};
    TestSizes     gemmOutMaxSizes = {1024, 1928};
    TestSizes     gemmOutMinSizes = {1024, 964};
    synGEMMParams gemmParams {};
    unsigned      gemmIn1 = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           gemmIn1MaxSizes.data(),
                                           2,
                                           syn_type_float,
                                           nullptr,
                                           "gemmIn2",
                                           0,
                                           0,
                                           nullptr,
                                           gemmIn1MinSizes.data());
    unsigned      gemmIn2 = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           gemmIn2MaxSizes.data(),
                                           2,
                                           syn_type_float);
    unsigned      gemmOut = createTensor(OUTPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    gemmOutMaxSizes.data(),
                                    2,
                                    syn_type_float,
                                    nullptr,
                                    gemmOutMinSizes.data());
    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {gemmIn1, gemmIn2},
                   {gemmOut},
                   &gemmParams,
                   sizeof(gemmParams),
                   "gemm");

    TestSizes reshapeShapeMaxSizes = {1024, 1928, 1};
    TestSizes reshapeShapeMinSizes = {1024, 964, 1};
    unsigned  reshapeShape =
        createShapeTensor(OUTPUT_TENSOR, reshapeShapeMaxSizes.data(), reshapeShapeMinSizes.data(), 3);
    unsigned reshapeOut = createTensor(OUTPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       reshapeShapeMaxSizes.data(),
                                       3,
                                       syn_type_float,
                                       nullptr,
                                       reshapeShapeMinSizes.data());
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {gemmOut, reshapeShape}, {reshapeOut});

    TestSizes addIn2Sizes    = {1024, 1, 1};
    TestSizes addOutMaxSizes = {1024, 1928, 1};
    TestSizes addOutMinSizes = {1024, 964, 1};
    unsigned  addIn2 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, addIn2Sizes.data(), 3, syn_type_float);
    unsigned addOut = createTensor(OUTPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   addOutMaxSizes.data(),
                                   3,
                                   syn_type_float,
                                   nullptr,
                                   addOutMinSizes.data());
    addNodeToGraph("add_fwd_f32", {addIn2, reshapeOut}, {addOut});

    compileTopology();
}

TEST_F_GC(SynGaudiConsumerReshapeAlignTest, tpc_dynamic_shape_consumer_reshape_align, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    TestSizes     gemmIn1MaxSizes = {1024, 1928};
    TestSizes     gemmIn1MinSizes = {1024, 964};
    TestSizes     gemmIn2MaxSizes = {1024, 1024};
    TestSizes     gemmOutMaxSizes = {1024, 1928};
    TestSizes     gemmOutMinSizes = {1024, 964};
    synGEMMParams gemmParams {};
    unsigned      gemmIn1 = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           gemmIn1MaxSizes.data(),
                                           2,
                                           syn_type_float,
                                           nullptr,
                                           "gemmIn2",
                                           0,
                                           0,
                                           nullptr,
                                           gemmIn1MinSizes.data());
    unsigned      gemmIn2 = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           gemmIn2MaxSizes.data(),
                                           2,
                                           syn_type_float);
    unsigned      gemmOut = createTensor(OUTPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    gemmOutMaxSizes.data(),
                                    2,
                                    syn_type_float,
                                    nullptr,
                                    gemmOutMinSizes.data());
    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {gemmIn1, gemmIn2},
                   {gemmOut},
                   &gemmParams,
                   sizeof(gemmParams),
                   "gemm");

    TestSizes reshapeShapeMaxSizes = {1024, 1928, 1};
    TestSizes reshapeShapeMinSizes = {1024, 964, 1};
    unsigned  reshapeShape =
        createShapeTensor(OUTPUT_TENSOR, reshapeShapeMaxSizes.data(), reshapeShapeMinSizes.data(), 3);
    unsigned reshapeOut = createTensor(OUTPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       reshapeShapeMaxSizes.data(),
                                       3,
                                       syn_type_float,
                                       nullptr,
                                       reshapeShapeMinSizes.data());
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {gemmOut, reshapeShape}, {reshapeOut});

    TestSizes addIn2MaxSizes = {1024, 1928, 1};
    TestSizes addIn2MinSizes = {1024, 964, 1};
    TestSizes addOutMaxSizes = {1024, 1928, 1};
    TestSizes addOutMinSizes = {1024, 964, 1};
    unsigned  addIn1         = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          addIn2MaxSizes.data(),
                                          3,
                                          syn_type_float,
                                          nullptr,
                                          "addIn2",
                                          0,
                                          0,
                                          nullptr,
                                          addIn2MinSizes.data());
    unsigned  addOut         = createTensor(OUTPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   addOutMaxSizes.data(),
                                   3,
                                   syn_type_float,
                                   nullptr,
                                   addOutMinSizes.data());
    addNodeToGraph("add_fwd_f32", {addIn1, reshapeOut}, {addOut});

    compileTopology();
}
