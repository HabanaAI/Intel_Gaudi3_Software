#include "gc_dynamic_shapes_infra.h"
#include "habana_global_conf.h"
#include "synapse_common_types.h"

class SynGaudiDynamicWideBucketTest : public SynGaudiDynamicShapesTestsInfra
{
private:
    bool m_prevWideBucket;

public:
    void SetUpTest() override
    {
        SynGaudiTestInfra::SetUpTest();
        m_prevWideBucket = GCFG_ENABLE_WIDE_BUCKET.value();
        GCFG_ENABLE_WIDE_BUCKET.setValue(true);
    }
    void TearDownTest() override
    {
        GCFG_ENABLE_WIDE_BUCKET.setValue(m_prevWideBucket);
        SynGaudiTestInfra::TearDownTest();
    }
};

TEST_F_GC(SynGaudiDynamicWideBucketTest, basic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned tMinSizes[] = {1023, 1023, 1, 1};
    unsigned tMaxSizes[] = {1024, 1024, 1, 1};
    unsigned tActSizes[] = {10, 10, 1, 1};  // actual is deliberately smaller than min
    unsigned tDim        = 4;

    unsigned tOutMinSizes[] = {1, 1023, 1, 1};
    unsigned tOutMaxSizes[] = {1, 1024, 1, 1};
    unsigned tOutActSizes[] = {1, 10, 1, 1};  // actual is deliberately smaller than min
    unsigned tOutDim        = 4;

    auto t1 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  tMaxSizes,
                                  tDim,
                                  syn_type_int32,
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
                                  syn_type_int32,
                                  nullptr,
                                  "t2",
                                  0,
                                  0,
                                  nullptr,
                                  tMinSizes);

    auto t3 = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ZERO,
                                  nullptr,
                                  tOutMaxSizes,
                                  tOutDim,
                                  syn_type_int32,
                                  nullptr,
                                  "t3",
                                  0,
                                  0,
                                  nullptr,
                                  tOutMinSizes);

    auto interm = createTensors(1,
                                OUTPUT_TENSOR,
                                false,  // isPersistent
                                "interm",
                                MEM_INIT_ALL_ZERO,
                                nullptr,  // initializer
                                tMaxSizes,
                                tDim,
                                syn_type_int32,
                                nullptr,
                                0,
                                0,
                                nullptr,
                                false,
                                tMinSizes)[0];

    addNodeToGraph("add_fwd_i32", {t1, t2}, {interm}, nullptr, 0);

    ns_Reduction::Params params;
    params.reductionDimension = 0;  // sum and reduce first dimension size to 1
    addNodeToGraph("reduce_sum_fwd_i32", {interm}, {t3}, &params, sizeof(params));

    compileTopology();
    setActualSizes(t1, tActSizes);
    setActualSizes(t2, tActSizes);
    setActualSizes(t3, tOutActSizes);

    int32_t* input1 = (int32_t*)m_hostBuffers[t1];
    int32_t* input2 = (int32_t*)m_hostBuffers[t2];
    int32_t* output = (int32_t*)m_hostBuffers[t3];
    memset(output, 0, tOutActSizes[1] * sizeof(float));

    runTopology();

    // validate the output
    //

    for (int i = 0; i < tOutActSizes[1]; ++i)
    {
        float sum = 0;
        for (int j = 0; j < tActSizes[0]; ++j)
        {
            sum += input1[i * tActSizes[0] + j] + input2[i * tActSizes[0] + j];
        }
        ASSERT_EQ(sum, output[i]) << "Wrong result, expected " << output[i] << " at index " << i;
    }
}
