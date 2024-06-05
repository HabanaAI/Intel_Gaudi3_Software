#include "gc_dynamic_shapes_infra.h"

class SynConstPersistentTensors : public SynGaudiTestInfra
{
};

TEST_F_GC(SynConstPersistentTensors, check_const_persistent_tensors_arent_allocated_by_infra_at_creation)
{
    // Testing the tests infra
    uint64_t    freeMemAtStart = 0;
    uint64_t    total          = 0;
    synDeviceId deviceId       = _getDeviceId();
    synStatus   status         = synDeviceGetMemoryInfo(deviceId, &freeMemAtStart, &total);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";

    constexpr unsigned dim              = 2;
    unsigned           in1DimSizes[]    = {8, 32};
    unsigned           in2DimSizes[]    = {16, 8};
    unsigned           outputDimSizes[] = {16, 32};

    const unsigned in1DataSize    = in1DimSizes[0] * in1DimSizes[1];
    const unsigned outputDataSize = outputDimSizes[0] * outputDimSizes[1];

    const auto    in1TensorIndex = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                    nullptr,
                                                    in1DimSizes,
                                                    dim,
                                                    syn_type_single,
                                                    nullptr,
                                                    "in1");
    const auto    in2TensorIndex = createConstPersistTensor(INPUT_TENSOR,
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         in2DimSizes,
                                                         dim,
                                                         syn_type_single,
                                                         nullptr,
                                                         "in2");
    const auto    outTensorIndex = createPersistTensor(OUTPUT_TENSOR,
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    outputDimSizes,
                                                    dim,
                                                    syn_type_single,
                                                    nullptr,
                                                    "out");
    unsigned char gemmParams[]   = {0, 0};
    addNodeToGraph("gemm", {in1TensorIndex, in2TensorIndex}, {outTensorIndex}, (void*)gemmParams, 2, "gemm");

    uint64_t freeMemBeforeCompile = 0;
    status                        = synDeviceGetMemoryInfo(deviceId, &freeMemBeforeCompile, &total);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";

    ASSERT_EQ(freeMemAtStart, freeMemBeforeCompile + (in1DataSize + outputDataSize) * sizeof(float))
        << "Expecting only in1 and output were allocated in the device";
}