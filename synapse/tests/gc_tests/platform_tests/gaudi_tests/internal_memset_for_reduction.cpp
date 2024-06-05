#include "gc_gaudi_test_infra.h"
TEST_F_GC(SynGaudiTestInfra, internal_memset_for_reduction, {synDeviceGaudi})
{
    unsigned in1MaxSize[] = {128, 131072};
    unsigned in1MinSize[] = {128, 131072};
    unsigned in1          = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "in1",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          in1MaxSize,
                                          2,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          in1MinSize,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned in2MaxSize[] = {131072};
    unsigned in2MinSize[] = {131072};
    unsigned in2          = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "in2",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          in2MaxSize,
                                          1,
                                          syn_type_int32,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          in2MinSize,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned in3MaxSize[] = {131072};
    unsigned in3MinSize[] = {131072};
    unsigned in3          = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "in3",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          in3MaxSize,
                                          1,
                                          syn_type_int32,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          in3MinSize,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned in4MaxSize[] = {128, 51008};
    unsigned in4MinSize[] = {128, 2};
    unsigned in4          = createTensors(1,
                                          INPUT_TENSOR,
                                          false,
                                          "in4",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          in4MaxSize,
                                          2,
                                          syn_type_uint32,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          in4MinSize,
                                          synTensorType::SHAPE_TENSOR)[0];

    unsigned outMaxSize[] = {128, 51008};
    unsigned outMinSize[] = {128, 2};
    unsigned out          = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "out",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          outMaxSize,
                                          2,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          outMinSize,
                                          synTensorType::DATA_TENSOR)[0];

    addNodeToGraph("sparse_segment_sum_fwd_f32", {in1, in2, in3, in4}, {out}, nullptr, 0, "SPARSE");

    // Memset will be executed on entire output, regardless of size.
    unsigned actualSizes[] = {128, 30000};
    setActualSizes(in4, actualSizes);
    setActualSizes(out, actualSizes);

    compileTopology("sparse_test_dyn", 0);
    runTopology(0 ,true);
    float * result = castHostInBuffer<float>(out);
    for (unsigned i = 0; i < actualSizes[0] * actualSizes[1]; i++)
    {
        ASSERT_EQ(result[i], 0);
    }
}
