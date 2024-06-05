#include "gc_dynamic_shapes_infra.h"
#include "synapse_common_types.h"
#include <algorithm>

class SynGaudiDynamicScatterNDONNXBigTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicScatterNDONNXBigTest, scatter, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    std::vector<int32_t> indicesInit(100);
    std::iota(indicesInit.begin(), indicesInit.end(), 0);
    std::random_shuffle(indicesInit.begin(), indicesInit.end());
    indicesInit.resize(200);    // to prevent heap buffer overflow

    unsigned indicesMinSizes[] = {1, 100};
    unsigned indicesMaxSizes[] = {1, 200};
    unsigned indicesActSizes[] = {1, 100};
    unsigned indicesDim        = 2;
    auto     indices           = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_FROM_INITIALIZER,
                                       reinterpret_cast<float*>(indicesInit.data()),
                                       indicesMaxSizes,
                                       indicesDim,
                                       syn_type_int32,
                                       nullptr,
                                       "indices",
                                       0,
                                       0,
                                       nullptr,
                                       indicesMinSizes);

    unsigned inputsMinSizes[] = {640, 426, 100};
    unsigned inputsMaxSizes[] = {1280, 966, 200};
    unsigned inputsActSizes[] = {640, 426, 100};
    unsigned inputsDim        = 3;
    auto     inputs           = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_POSITIVE,
                                      nullptr,
                                      inputsMaxSizes,
                                      inputsDim,
                                      syn_type_float,
                                      nullptr,
                                      "inputs",
                                      0,
                                      0,
                                      nullptr,
                                      inputsMinSizes);

    unsigned updatesMinSizes[] = {640, 426, 100};
    unsigned updatesMaxSizes[] = {1280, 966, 200};
    unsigned updatesActSizes[] = {640, 426, 100};
    unsigned updatesDim        = 3;
    auto     updates           = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       updatesMaxSizes,
                                       updatesDim,
                                       syn_type_float,
                                       nullptr,
                                       "updates",
                                       0,
                                       0,
                                       nullptr,
                                       updatesMinSizes);

    unsigned outputsMinSizes[] = {640, 426, 100};
    unsigned outputsMaxSizes[] = {1280, 966, 200};
    unsigned outputsActSizes[] = {640, 426, 100};
    unsigned outputsDim        = 3;
    auto     outputs           = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ONES,
                                       nullptr,
                                       outputsMaxSizes,
                                       outputsDim,
                                       syn_type_float,
                                       nullptr,
                                       "outputs",
                                       0,
                                       0,
                                       nullptr,
                                       outputsMinSizes);

    addNodeToGraph("scatter_nd_onnx_fwd_f32", {inputs, indices, updates}, {outputs}, nullptr, 0);

    compileTopology();
    setActualSizes(inputs, inputsActSizes);
    setActualSizes(indices, indicesActSizes);
    setActualSizes(updates, updatesActSizes);
    setActualSizes(outputs, outputsActSizes);
    runTopology();

    auto* indicesBuffer = castHostInBuffer<int32_t>(indices);
    auto* inputsBuffer  = castHostInBuffer<float>(inputs);
    auto* updatesBuffer = castHostInBuffer<float>(updates);
    auto* outputsBuffer = castHostOutBuffer<float>(outputs);

    // calculate the expected outputs
    std::vector<float> expectedOutputs((std::size_t)(outputsActSizes[0] * outputsActSizes[1] * outputsActSizes[2]));
    for (unsigned i = 0; i < expectedOutputs.size(); ++i)
        expectedOutputs[i] = inputsBuffer[i];
    for (unsigned i = 0; i < indicesActSizes[1]; ++i)
    {
        for (unsigned j = 0; j < inputsActSizes[1]; ++j)
        {
            for (unsigned k = 0; k < inputsActSizes[0]; ++k)
            {
                unsigned outputsIndex =
                    k + j * inputsActSizes[0] + indicesBuffer[i] * inputsActSizes[0] * inputsActSizes[1];
                unsigned updatesIndex = k + j * outputsActSizes[0] + i * outputsActSizes[0] * outputsActSizes[1];
                expectedOutputs[outputsIndex] = updatesBuffer[updatesIndex];
            }
        }
    }

    // compare expected with actual
    for (unsigned i = 0; i < expectedOutputs.size(); i++)
    {
        ASSERT_EQ(outputsBuffer[i], expectedOutputs[i]) << " at index " << i;
    }
}
