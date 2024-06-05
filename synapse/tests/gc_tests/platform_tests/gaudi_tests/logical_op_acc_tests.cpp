#include "node_factory.h"
#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"

class SynTrainingLogicalOpAccTest : public SynTrainingTestInfra
{
};

TEST_F_GC(SynTrainingLogicalOpAccTest, concat_transpose_reshape_seq)
{
    const unsigned BATCH = 2;
    const unsigned HEIGHT = 20;
    const unsigned WIDTH = 30;
    const unsigned CHANNEL = 50;
    TestSizes concatInSize = {CHANNEL, WIDTH, HEIGHT, 1, 1};
    TestSizes concatOutSize = {CHANNEL, WIDTH, HEIGHT, BATCH, 1};
    TestSizes transposeSize = {CHANNEL, HEIGHT, WIDTH, BATCH, 1};
    TestSizes reshapeSize = {CHANNEL * WIDTH, HEIGHT * BATCH, 1, 1, 1};

    unsigned reluIn1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, concatInSize.data(), 4, syn_type_float, nullptr, "reluIn1");
    unsigned reluIn2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, concatInSize.data(), 4, syn_type_float, nullptr, "reluIn2");
    unsigned reluOut1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, concatInSize.data(), 4, syn_type_float);
    unsigned reluOut2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, concatInSize.data(), 4, syn_type_float);
    unsigned concatOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, concatOutSize.data(), 4, syn_type_float);
    unsigned transposeOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, transposeSize.data(), 4, syn_type_float);
    unsigned reshapeOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, reshapeSize.data(), 2, syn_type_float);
    unsigned final = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, reshapeSize.data(), 2, syn_type_float, nullptr, "final");

    addNodeToGraph("relu_fwd_f32", {reluIn1}, {reluOut1}, nullptr, 0, "relu1");
    addNodeToGraph("relu_fwd_f32", {reluIn2}, {reluOut2}, nullptr, 0, "relu2");
    unsigned concatDim = 3;
    addNodeToGraph(NodeFactory::concatenateNodeTypeName, {reluOut1, reluOut2}, {concatOut}, &concatDim, sizeof(concatDim), "concat");
    synTransposeParams transposeParam;
    transposeParam.permutation[0] = TPD_Channel;
    transposeParam.permutation[1] = TPD_Height;
    transposeParam.permutation[2] = TPD_Width;
    transposeParam.permutation[3] = TPD_4Dim_Batch;
    transposeParam.tensorDim = 4;
    addNodeToGraph(NodeFactory::transposeNodeTypeName, {concatOut}, {transposeOut}, &transposeParam, sizeof(transposeParam), "transpose");
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {transposeOut}, {reshapeOut}, nullptr, 0, "reshape");
    addNodeToGraph("relu_fwd_f32", {reshapeOut}, {final}, nullptr, 0, "reluFinal");

    compileAndRun();
    float* out = castHostBuffer<float>(final);
    for (unsigned batch = 0; batch  < BATCH; ++batch)
    {
        float* in = castHostBuffer<float>(batch == 0? reluIn1 : reluIn2);
        for (unsigned height = 0; height  < HEIGHT; ++height)
        {
            for (unsigned width = 0; width  < WIDTH; ++width)
            {
                for (unsigned channel = 0; channel  < CHANNEL; ++channel)
                {
                    unsigned inIdx = channel + width * CHANNEL + height * CHANNEL * WIDTH;
                    unsigned outIdx = channel + height * CHANNEL + width * CHANNEL * HEIGHT + batch * CHANNEL * WIDTH * HEIGHT;
                    ASSERT_EQ(out[outIdx], in[inIdx]) << "Wrong value at [" << channel << "," << width << "," << height << "," << batch << "]";
                }
            }
        }
    }
}

TEST_F_GC(SynTrainingLogicalOpAccTest, logical_transpose_relu_big_tensor_ASIC, {synDeviceGaudi, synDeviceGaudi2})
{
    const unsigned BATCH1 = 1<<26;
    const unsigned BATCH2 = 2;
    const unsigned HEIGHT = 1;
    const unsigned WIDTH = 2;
    const unsigned CHANNEL = 4;
    TestSizes transposeInSize = {CHANNEL, WIDTH, HEIGHT, BATCH2, BATCH1};
    TestSizes transpoeOutSize = {CHANNEL, WIDTH, HEIGHT, BATCH1, BATCH2};

    unsigned transposeIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE_ONLY, nullptr, transposeInSize.data(), 5, syn_type_float, nullptr, "transposeIn");
    unsigned transposeOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, transpoeOutSize.data(), 5, syn_type_float);
    unsigned final = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, transpoeOutSize.data(), 5, syn_type_float, nullptr, "final");

    synTransposeParams transposeParam;
    transposeParam.permutation[0] = TPD_Channel;
    transposeParam.permutation[1] = TPD_Width;
    transposeParam.permutation[2] = TPD_Height;
    transposeParam.permutation[3] = TPD_Batch;
    transposeParam.permutation[4] = TPD_4Dim_Batch;
    transposeParam.tensorDim = 5;
    addNodeToGraph(NodeFactory::transposeNodeTypeName, {transposeIn}, {transposeOut}, &transposeParam, sizeof(transposeParam), "transpose");
    addNodeToGraph("relu_fwd_f32", {transposeOut}, {final}, nullptr, 0, "reluFinal");

    compileAndRun();
    float* out = castHostBuffer<float>(final);
    for (unsigned batch = 0; batch  < BATCH1*BATCH2; ++batch)
    {
        for (unsigned height = 0; height  < HEIGHT; ++height)
        {
            for (unsigned width = 0; width  < WIDTH; ++width)
            {
                for (unsigned channel = 0; channel  < CHANNEL; ++channel)
                {
                    ASSERT_EQ(*out, 0) << "Wrong value at [" << channel << "," << width << "," << height << "," << batch << "]";
                    out++;
                }
            }
        }
    }
}
