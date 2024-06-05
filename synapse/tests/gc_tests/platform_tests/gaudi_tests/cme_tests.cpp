#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "timer.h"
#include <tuple>
#include <deque>
#include "tensor_io_indicies.h"
#include "scoped_configuration_change.h"
#include "infra/global_conf_manager.h"

class SynTrainingTestInfraCmePerf : public SynTrainingTestInfra
{
};

TEST_F_GC(SynTrainingTestInfraCmePerf, long_parallel_graph, {synDeviceGaudi3})
{
    ScopedConfigurationChange experimentalFlagsCfg("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange skipBundleCheckCfg("LITE_PERFORATION_SKIP_BUNDLE_CHECK", "true");
    ScopedConfigurationChange disableTpcFuserCfg("RUN_TPC_FUSER", "false");

    static const unsigned numNodes = 300;

    unsigned reluIn1  = 0;
    unsigned reluOut1 = 0;
    unsigned reluIn2  = 0;
    unsigned reluOut2 = 0;
    unsigned reluIn3  = 0;
    unsigned reluOut3 = 0;

    unsigned rollingIdxIn1  = 0;
    unsigned rollingIdxOut1 = 0;

    unsigned rollingIdxIn2  = 0;
    unsigned rollingIdxOut2 = 0;

    unsigned rollingIdxIn3  = 0;
    unsigned rollingIdxOut3 = 0;

    // create data - 8MB tensors
    unsigned dims    = 4;
    unsigned sizes[] = {1000, 1, 16, 64};

    // Graph havs this structure: [relu_fwd]->[relu_fwd]->[relu_fwd]->[relu_fwd]->[relu_fwd]-> ... ->[relu_fwd]
    // We have 3 long graphs running in parallel. We want to use the entire cache (96MB). There is total
    // of (3 * 300 = 900) nodes and a DISCARD operation will be generated for each output tensor (8MB) ~= 7200MB
    reluIn1        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, dims);
    rollingIdxOut1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    reluIn2        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, dims);
    rollingIdxOut2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    reluIn3        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, dims);
    rollingIdxOut3 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    addNodeToGraph("relu_fwd_f32", {reluIn1}, {rollingIdxOut1});
    addNodeToGraph("relu_fwd_f32", {reluIn2}, {rollingIdxOut2});
    addNodeToGraph("relu_fwd_f32", {reluIn3}, {rollingIdxOut3});

    for (unsigned i = 0; i < numNodes; i++)
    {
        rollingIdxIn1  = connectOutputTensorToInputTensor(rollingIdxOut1);
        rollingIdxOut1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

        rollingIdxIn2  = connectOutputTensorToInputTensor(rollingIdxOut2);
        rollingIdxOut2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

        rollingIdxIn3  = connectOutputTensorToInputTensor(rollingIdxOut3);
        rollingIdxOut3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

        addNodeToGraph("relu_fwd_f32", {rollingIdxIn1}, {rollingIdxOut1});
        addNodeToGraph("relu_fwd_f32", {rollingIdxIn2}, {rollingIdxOut2});
        addNodeToGraph("relu_fwd_f32", {rollingIdxIn3}, {rollingIdxOut3});
    }
    rollingIdxIn1 = connectOutputTensorToInputTensor(rollingIdxOut1);
    reluOut1      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    rollingIdxIn2 = connectOutputTensorToInputTensor(rollingIdxOut2);
    reluOut2      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    rollingIdxIn3 = connectOutputTensorToInputTensor(rollingIdxOut3);
    reluOut3      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    addNodeToGraph("relu_fwd_f32", {rollingIdxIn1}, {reluOut1});
    addNodeToGraph("relu_fwd_f32", {rollingIdxIn2}, {reluOut2});
    addNodeToGraph("relu_fwd_f32", {rollingIdxIn3}, {reluOut3});

    compileTopology();

    runTopology();

    // validate results
    float* inputBuffer1  = castHostInBuffer<float>(reluIn1);
    float* outputBuffer1 = castHostOutBuffer<float>(reluOut1);
    float* inputBuffer2  = castHostInBuffer<float>(reluIn2);
    float* outputBuffer2 = castHostOutBuffer<float>(reluOut2);
    float* inputBuffer3  = castHostInBuffer<float>(reluIn3);
    float* outputBuffer3 = castHostOutBuffer<float>(reluOut3);

    for (uint64_t i = 0; i < getTensorElementCount(reluIn1); ++i)
    {
        ASSERT_EQ(inputBuffer1[i], outputBuffer1[i]);
        ASSERT_EQ(inputBuffer2[i], outputBuffer2[i]);
        ASSERT_EQ(inputBuffer3[i], outputBuffer3[i]);
    }
}

class SynTrainingTestInfraCme : public SynTrainingTestInfra
{
public:
    ~SynTrainingTestInfraCme() { ReleaseDevice(); }

    void SetUpTest()
    {
        SynTest::SetUpTest();

        if (shouldRunTest())
        {
            // Set the mcid discard limit just before the graph is constructed
            GlobalConfManager::instance().setGlobalConf("CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING", "156");
            init();
        }
    }
};

TEST_F_GC(SynTrainingTestInfraCme, long_graph_with_multiple_rollover, {synDeviceGaudi3})
{
    ScopedConfigurationChange experimentalFlagsCfg("ENABLE_EXPERIMENTAL_FLAGS", "true");
    ScopedConfigurationChange skipBundleCheckCfg("LITE_PERFORATION_SKIP_BUNDLE_CHECK", "true");
    ScopedConfigurationChange disableTpcFuserCfg("RUN_TPC_FUSER", "false");

    static const unsigned numNodes = 100;

    unsigned reluIn1  = 0;
    unsigned reluOut1 = 0;
    unsigned reluIn2  = 0;
    unsigned reluOut2 = 0;
    unsigned reluIn3  = 0;
    unsigned reluOut3 = 0;

    unsigned rollingIdxIn1  = 0;
    unsigned rollingIdxOut1 = 0;

    unsigned rollingIdxIn2  = 0;
    unsigned rollingIdxOut2 = 0;

    unsigned rollingIdxIn3  = 0;
    unsigned rollingIdxOut3 = 0;

    unsigned dims    = 4;
    unsigned sizes[] = {1000, 1, 4, 16};

    // Graph havs this structure: [relu_fwd]->[relu_fwd]->[relu_fwd]->[relu_fwd]->[relu_fwd]-> ... ->[relu_fwd]
    // We are going to perform multiple MCID rollover
    reluIn1        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, dims);
    rollingIdxOut1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    reluIn2        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, dims);
    rollingIdxOut2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    reluIn3        = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, dims);
    rollingIdxOut3 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    addNodeToGraph("relu_fwd_f32", {reluIn1}, {rollingIdxOut1});
    addNodeToGraph("relu_fwd_f32", {reluIn2}, {rollingIdxOut2});
    addNodeToGraph("relu_fwd_f32", {reluIn3}, {rollingIdxOut3});

    for (unsigned i = 0; i < numNodes; i++)
    {
        rollingIdxIn1  = connectOutputTensorToInputTensor(rollingIdxOut1);
        rollingIdxOut1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

        rollingIdxIn2  = connectOutputTensorToInputTensor(rollingIdxOut2);
        rollingIdxOut2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

        rollingIdxIn3  = connectOutputTensorToInputTensor(rollingIdxOut3);
        rollingIdxOut3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

        addNodeToGraph("relu_fwd_f32", {rollingIdxIn1}, {rollingIdxOut1});
        addNodeToGraph("relu_fwd_f32", {rollingIdxIn2}, {rollingIdxOut2});
        addNodeToGraph("relu_fwd_f32", {rollingIdxIn3}, {rollingIdxOut3});
    }
    rollingIdxIn1 = connectOutputTensorToInputTensor(rollingIdxOut1);
    reluOut1      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    rollingIdxIn2 = connectOutputTensorToInputTensor(rollingIdxOut2);
    reluOut2      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    rollingIdxIn3 = connectOutputTensorToInputTensor(rollingIdxOut3);
    reluOut3      = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, dims);

    addNodeToGraph("relu_fwd_f32", {rollingIdxIn1}, {reluOut1});
    addNodeToGraph("relu_fwd_f32", {rollingIdxIn2}, {reluOut2});
    addNodeToGraph("relu_fwd_f32", {rollingIdxIn3}, {reluOut3});

    compileTopology();

    runTopology();

    // validate results
    float* inputBuffer1  = castHostInBuffer<float>(reluIn1);
    float* outputBuffer1 = castHostOutBuffer<float>(reluOut1);
    float* inputBuffer2  = castHostInBuffer<float>(reluIn2);
    float* outputBuffer2 = castHostOutBuffer<float>(reluOut2);
    float* inputBuffer3  = castHostInBuffer<float>(reluIn3);
    float* outputBuffer3 = castHostOutBuffer<float>(reluOut3);

    for (uint64_t i = 0; i < getTensorElementCount(reluIn1); ++i)
    {
        ASSERT_EQ(inputBuffer1[i], outputBuffer1[i]);
        ASSERT_EQ(inputBuffer2[i], outputBuffer2[i]);
        ASSERT_EQ(inputBuffer3[i], outputBuffer3[i]);
    }
}
