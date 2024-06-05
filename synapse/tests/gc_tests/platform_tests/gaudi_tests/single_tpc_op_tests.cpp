#include "gc_dynamic_shapes_infra.h"
#include "synapse_common_types.h"

class SynGaudiDynamicSingleOpTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicSingleOpTest, single_tpc_op_for_profiling_ASIC_CI, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned tActSizes[] = {32, 3, 320, 672};
    unsigned tMinSizes[] = {25, 2, 294, 659};
    unsigned tMaxSizes[] = {64, 6, 640, 1344};

    unsigned tDim = 4;

    auto t1 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_single,
                                  nullptr,
                                  "t1",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);

    auto t2 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_single,
                                  nullptr,
                                  "t2",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);

    auto t3 = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ZERO,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_single,
                                  nullptr,
                                  "t3",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);

    addNodeToGraph("mult_fwd_f32", {t1, t2}, {t3}, nullptr, 0);

    compileTopology();
    setActualSizes(t1, tActSizes);
    setActualSizes(t2, tActSizes);
    setActualSizes(t3, tActSizes);

    float* input1 = (float*)m_hostBuffers[t1];
    float* input2 = (float*)m_hostBuffers[t2];
    float* output = (float*)m_hostBuffers[t3];
    memset(output, 0, tActSizes[1] * sizeof(float));

    runTopology();

    // validate the output
    //

    for (int i = 0; i < tActSizes[0] * tActSizes[1] * tActSizes[2] * tActSizes[3]; ++i)
    {
        ASSERT_FLOAT_EQ(input1[i] * input2[i], output[i])
            << "Wrong result, expected " << output[i] << " at index " << i;
    }
}
