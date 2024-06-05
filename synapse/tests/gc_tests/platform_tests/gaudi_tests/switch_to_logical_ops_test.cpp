#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "tensor.h"
#include "tensor_validator.inl"
#include "test_utils.h"
#include "utils.h"
#include "gc_tests/unit_tests/gaudi/tile_utils.hpp"

class SynTrainingTileTest
: public SynTrainingTestInfra
, public TileTestUtilsBase<TestSizeVec>
{
};

TEST_F_GC(SynTrainingTestInfra, tile_to_broadcast_simple)
{
    const unsigned n       = 1;
    const unsigned nTile   = 8;
    const unsigned nOutput = n * nTile;
    const unsigned nStride = (nTile == 1) ? 1 : 0;

    const unsigned h       = 1;
    const unsigned hTile   = 3;
    const unsigned hOutput = h * hTile;
    const unsigned hStride = (hTile == 1) ? 1 : 0;

    const unsigned w       = 16;
    const unsigned wTile   = 1;
    const unsigned wOutput = w * wTile;
    const unsigned wStride = (wTile == 1) ? 1 : 0;

    const unsigned c       = 16;
    const unsigned cTile   = 1;
    const unsigned cOutput = c * cTile;

    unsigned inputSizes[]  = {c, w, h, n};
    unsigned outputSizes[] = {cOutput, wOutput, hOutput, nOutput};

    auto in  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inputSizes);
    auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes);

    ns_TileKernel::ParamsV2 params {.repeat = {cTile, wTile, hTile, nTile}};
    addNodeToGraph("tile_fwd_f32", &params, sizeof(params));
    compileAndRun();

    float* pInputBuffer  = (float*)m_hostBuffers[in];
    float* pOutputBuffer = (float*)m_hostBuffers[out];

    for (int l = 0; l < nOutput; l++)
    {
        for (int k = 0; k < hOutput; k++)
        {
            for (int i = 0; i < wOutput; i++)
            {
                for (int j = 0; j < cOutput; j++)
                {
                    auto output =
                        pOutputBuffer[j + (i * cOutput) + (k * cOutput * wOutput) + (l * cOutput * wOutput * hOutput)];
                    auto input =
                        pInputBuffer[j + (i * cOutput * wStride) + (k * c * w * hStride) + (l * c * w * h * nStride)];

                    ASSERT_EQ(input, output) << "Mismatch for at index " << j << i << k << l << " Expected:" << input
                                             << " Result: " << output;
                }
            }
        }
    }
}

TEST_P_GC(SynTrainingTileTest, tile_to_broadcast_f32)
{
    auto inputSizes  = ::testing::get<0>(GetParam());
    auto outputSizes = ::testing::get<1>(GetParam());

    uint8_t inputDim  = inputSizes.size();
    uint8_t outputDim = outputSizes.size();

    std::fill_n(std::back_inserter(inputSizes), SYN_MAX_TENSOR_DIM - inputDim, 0);
    std::fill_n(std::back_inserter(outputSizes), SYN_MAX_TENSOR_DIM - outputDim, 0);

    auto in1_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, inputSizes.data(), inputDim);
    auto in1_2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, inputSizes.data(), inputDim);
    auto out1  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inputSizes.data(), inputDim);

    auto in2_1 = connectOutputTensorToInputTensor(out1);
    auto out2  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes.data(), outputDim);

    auto in3_1 = connectOutputTensorToInputTensor(out2);
    auto out3  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes.data(), outputDim);

    addNodeToGraph("add_fwd_f32", {in1_1, in1_2}, {out1}, nullptr, 0);

    ns_TileKernel::ParamsV2 params;
    calculateTileParams(inputSizes, outputSizes, params);

    addNodeToGraph("tile_fwd_f32", {in2_1}, {out2}, &params, sizeof(params));
    addNodeToGraph("relu_fwd_f32", {in3_1}, {out3}, nullptr, 0, "relu");

    compileAndRun();

    float* pInputBuffer1 = (float*)m_hostBuffers[in1_1];
    float* pInputBuffer2 = (float*)m_hostBuffers[in1_2];
    float* pOutputBuffer = (float*)m_hostBuffers[out3];

    const unsigned& c = inputSizes[0];
    const unsigned& w = inputSizes[1];
    const unsigned& h = inputSizes[2];
    const unsigned& n = inputSizes[3];

    const unsigned cOutput = outputSizes[0];
    const unsigned wOutput = (outputSizes[1] == 0) ? 1 : outputSizes[1];
    const unsigned hOutput = (outputSizes[2] == 0) ? 1 : outputSizes[2];
    const unsigned nOutput = (outputSizes[3] == 0) ? 1 : outputSizes[3];
    const unsigned bOutput = (outputSizes[4] == 0) ? 1 : outputSizes[4];

    const unsigned& wTile = params.repeat[1];
    const unsigned& hTile = params.repeat[2];
    const unsigned& nTile = params.repeat[3];
    const unsigned& bTile = params.repeat[4];

    const unsigned bStride = (bTile == 1) ? 1 : 0;
    const unsigned nStride = (nTile == 1) ? 1 : 0;
    const unsigned hStride = (hTile == 1) ? 1 : 0;
    const unsigned wStride = (wTile == 1) ? 1 : 0;

    unsigned int actualCount = 0;

    for (int m = 0; m < bOutput; m++)
    {
        for (int l = 0; l < nOutput; l++)
        {
            for (int k = 0; k < hOutput; k++)
            {
                for (int i = 0; i < wOutput; i++)
                {
                    for (int j = 0; j < cOutput; j++)
                    {
                        auto index = j + (i * cOutput * wStride) + (k * c * w * hStride) + (l * c * w * h * nStride) +
                                     (m * c * w * h * n * bStride);
                        auto output = pOutputBuffer[j + (i * cOutput) + (k * cOutput * wOutput) +
                                                    (l * cOutput * wOutput * hOutput) +
                                                    (m * cOutput * wOutput * hOutput * nOutput)];
                        auto input  = std::max((float)0.0, pInputBuffer1[index] + pInputBuffer2[index]);

                        actualCount++;

                        ASSERT_EQ(input, output) << "Mismatch for at index " << j << i << k << l << m
                                                 << " Expected:" << input << " Result: " << output;
                    }
                }
            }
        }
    }
    uint32_t elementsCount =
        std::accumulate(std::begin(outputSizes), std::begin(outputSizes) + outputDim, 1, std::multiplies<int>());

    ASSERT_EQ(actualCount, elementsCount)
        << "Element count mismatch: actual " << actualCount << " expected: " << elementsCount;
}

TEST_F_GC(SynTrainingTestInfra, tile_to_broadcast_bf16)
{
    const unsigned N        = 1;
    const unsigned N_tile   = 16;
    const unsigned N_output = N * N_tile;
    const unsigned N_stride = (N_tile == 1) ? 1 : 0;

    const unsigned H        = 16;
    const unsigned H_tile   = 1;
    const unsigned H_output = H * H_tile;
    const unsigned H_stride = (H_tile == 1) ? 1 : 0;

    const unsigned W        = 16;
    const unsigned W_tile   = 1;
    const unsigned W_output = W * W_tile;
    const unsigned W_stride = (W_tile == 1) ? 1 : 0;

    const unsigned C        = 16;
    const unsigned C_tile   = 1;
    const unsigned C_output = C * C_tile;

    unsigned inputSizes[]  = {C, W, H, N};
    unsigned outputSizes[] = {C_output, W_output, H_output, N_output};

    auto in1_1 = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_POSITIVE,
                                     nullptr,
                                     inputSizes,
                                     ARRAY_SIZE(inputSizes),
                                     syn_type_bf16);
    auto in1_2 = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_RANDOM_POSITIVE,
                                     nullptr,
                                     inputSizes,
                                     ARRAY_SIZE(inputSizes),
                                     syn_type_bf16);
    auto out1 =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inputSizes, ARRAY_SIZE(inputSizes), syn_type_bf16);

    auto in2_1 = connectOutputTensorToInputTensor(out1);
    auto out2 =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, ARRAY_SIZE(outputSizes), syn_type_bf16);

    auto in3_1 = connectOutputTensorToInputTensor(out2);
    auto out3  = createPersistTensor(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    outputSizes,
                                    ARRAY_SIZE(outputSizes),
                                    syn_type_bf16);

    addNodeToGraph("add_fwd_bf16", {in1_1, in1_2}, {out1}, nullptr, 0);
    ns_TileKernel::ParamsV2 params {.repeat = {C_tile, W_tile, H_tile, N_tile}};
    addNodeToGraph("tile_fwd_bf16", {in2_1}, {out2}, &params, sizeof(params));
    addNodeToGraph("relu_fwd_bf16", {in3_1}, {out3}, nullptr, 0, "relu");

    compileAndRun();

    bfloat16* pInputBuffer1 = castHostInBuffer<bfloat16>(in1_1);
    bfloat16* pInputBuffer2 = castHostInBuffer<bfloat16>(in1_2);
    bfloat16* pOutputBuffer = castHostOutBuffer<bfloat16>(out3);

    unsigned totalOutputSize = C_output * W_output * H_output * N_output;

    std::vector<float> expectedResult(totalOutputSize, -1.0);

    for (int l = 0; l < N_output; l++)
    {
        for (int k = 0; k < H_output; k++)
        {
            for (int i = 0; i < W_output; i++)
            {
                for (int j = 0; j < C_output; j++)
                {
                    auto inputIndex =
                        j + (i * C_output * W_stride) + (k * C * W * H_stride) + (l * C * W * H * N_stride);
                    auto outputIndex =
                        j + (i * C_output) + (k * C_output * W_output) + (l * C_output * W_output * H_output);
                    expectedResult[outputIndex] =
                        std::max((float)0.0, float(pInputBuffer1[inputIndex]) + float(pInputBuffer2[inputIndex]));
                }
            }
        }
    }

    validateResult(expectedResult.data(), pOutputBuffer, totalOutputSize);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SynTrainingTileTest,
    ::testing::Values(::testing::make_tuple(TestSizeVec {4, 1}, TestSizeVec {4, 4}),
                      ::testing::make_tuple(TestSizeVec {4, 1}, TestSizeVec {4, 4, 5}),
                      ::testing::make_tuple(TestSizeVec {2, 1, 2, 2}, TestSizeVec {2, 2, 2, 2}),
                      ::testing::make_tuple(TestSizeVec {2, 1, 1, 1}, TestSizeVec {2, 3, 4, 5}),
                      ::testing::make_tuple(TestSizeVec {2, 1, 2, 1}, TestSizeVec {2, 3, 2, 3}),
                      ::testing::make_tuple(TestSizeVec {3, 1, 1, 1}, TestSizeVec {3, 2, 2, 2, 2}),
                      ::testing::make_tuple(TestSizeVec {3, 1, 1, 1, 1}, TestSizeVec {3, 2, 2, 2, 2}),
                      ::testing::make_tuple(TestSizeVec {1, 2, 1, 4, 1}, TestSizeVec {1, 2, 3, 4, 5}),
                      ::testing::make_tuple(TestSizeVec {16, 16, 16, 1}, TestSizeVec {16, 16, 16, 16}),
                      ::testing::make_tuple(TestSizeVec {16, 16, 1}, TestSizeVec {16, 16, 16, 16})),
    SynTrainingTileTest::GetName());