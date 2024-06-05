#include "gc_dynamic_shapes_infra.h"

// TODO: add mme memset test.

class SynGaudiDynamicReduction : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicReduction, basic)
{
    unsigned inMaxSizes[]  = {8, 8192};
    unsigned inMinSizes[]  = {4, 4};
    unsigned outMaxSizes[] = {8, 1};
    unsigned outMinSizes[] = {4, 1};
    unsigned inActSizes[]  = {6, 12};
    unsigned outActSizes[] = {6, 1};

    auto ti  = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  inMaxSizes,
                                  2,
                                  syn_type_single,
                                  nullptr,
                                  "ti",
                                  0,
                                  0,
                                  nullptr,
                                  inMinSizes);
    auto to1 = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_NONE,
                                   nullptr,
                                   outMaxSizes,
                                   2,
                                   syn_type_single,
                                   nullptr,
                                   "to1",
                                   0,
                                   0,
                                   nullptr,
                                   outMinSizes);
    auto to2 = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_NONE,
                                   nullptr,
                                   outMaxSizes,
                                   2,
                                   syn_type_int32,
                                   nullptr,
                                   "to2",
                                   0,
                                   0,
                                   nullptr,
                                   outMinSizes);

    ns_Reduction::Params nodeParams {1};  // reduce along dim 1

    addNodeToGraph("reduce_max_fwd_f32", {ti}, {to1, to2}, (void*)&nodeParams, sizeof(nodeParams));

    setActualSizes(ti, inActSizes);
    setActualSizes(to1, outActSizes);
    setActualSizes(to2, outActSizes);

    compileTopology();
    runTopology();

    float* inBuffer       = castHostBuffer<float>(ti);
    float* outBuffer      = castHostBuffer<float>(to1);
    int*   outIndexBuffer = castHostBuffer<int>(to2);

    for (int i = 0; i < inActSizes[0]; ++i)
    {
        int   maxIdx = -1;
        float maxVal = -1;
        for (int j = 0; j < inActSizes[1]; ++j)
        {
            if (inBuffer[inActSizes[0] * j + i] > maxVal)
            {
                maxVal = inBuffer[inActSizes[0] * j + i];
                maxIdx = j;
            }
        }

        ASSERT_EQ(outBuffer[i], maxVal) << "Mismatch at i=" << i << " maxVal expected " << maxVal << " actual "
                                        << outBuffer[i];
        ASSERT_EQ(outIndexBuffer[i], maxIdx)
            << "Mismatch at i=" << i << " maxIdx expected " << maxIdx << " actual " << outIndexBuffer[i];
    }
}
