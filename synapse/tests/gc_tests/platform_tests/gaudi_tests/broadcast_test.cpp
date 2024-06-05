#include "gc_gaudi_test_infra.h"
#include "gc_dynamic_shapes_infra.h"
#include "global_conf_test_setter.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "syn_singleton.hpp"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>
#include "syn_gaudi_two_run_compare_test.h"
#include "scoped_configuration_change.h"

class SynTrainingBroadcastTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<std::tuple<unsigned, unsigned>,  // tensor dim, broadcast mask
                                                unsigned,                        // dim 0 size
                                                unsigned,                        // dim 1 size
                                                unsigned,                        // dim 2 size
                                                unsigned,                        // dim 3 size
                                                unsigned>>                       // dim 4 size
{
public:
    SynTrainingBroadcastTest() { setTestPackage(TEST_PACKAGE_BROADCAST); }

protected:
    static const unsigned PRIME = 7919;

    void run(const TestSizeVec& sizesIn, const TestSizeVec& sizesOut)
    {
        std::array<TestSizeVec, 3> sizesIn2  = {sizesIn, sizesIn, sizesIn};
        std::array<TestSizeVec, 3> sizesOut2 = {sizesOut, sizesOut, sizesOut};
        run(sizesIn2, sizesOut2);
    }

    void run(std::array<TestSizeVec, 3>& sizesIn, std::array<TestSizeVec, 3>& sizesOut)
    {
        unsigned      tensorDim      = sizesOut[0].size();
        bool          isDynamicShape = sizesOut[0] != sizesOut[1];
        TensorIndices inputs;
        unsigned      input  = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             sizesIn[0].data(),
                                             tensorDim,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             sizesIn[1].data());
        unsigned      output = createPersistTensor(OUTPUT_TENSOR,
                                              MEM_INIT_ALL_ONES,
                                              nullptr,
                                              sizesOut[0].data(),
                                              tensorDim,
                                              syn_type_float,
                                              nullptr,
                                              nullptr,
                                              0,
                                              0,
                                              nullptr,
                                              sizesOut[1].data());

        inputs.push_back(input);
        if (isDynamicShape)
        {
            unsigned shape = createShapeTensor(INPUT_TENSOR, sizesOut[0].data(), sizesOut[1].data(), tensorDim);
            inputs.push_back(shape);
        }

        addNodeToGraph("broadcast", inputs, {output}, nullptr, 0, "broadcast");

        compileTopology();
        if (!HasFailure())
        {
            if (isDynamicShape)
            {
                setActualSizes(input, sizesIn[2]);
                setActualSizes(output, sizesOut[2]);
                setActualSizes(inputs.back(), sizesOut[2]);
            }
            runTopology(0, true);
            float* pOutputBuffer = (float*)m_hostBuffers[output];
            float* pInputBuffer  = (float*)m_hostBuffers[input];

            validateResults(pOutputBuffer, pInputBuffer, sizesIn, sizesOut);
        }
    }

    static inline bool shouldBeBroadcastDim(const unsigned mask, const unsigned dim) { return (1 << dim) & mask; }

private:
    void validateResults(const float*               pOutputBuffer,
                         const float*               pInputBuffer,
                         std::array<TestSizeVec, 3>& sizesIn,
                         std::array<TestSizeVec, 3>& sizesOut,
                         unsigned                   dim          = 0,
                         unsigned                   inIndex      = 0,
                         unsigned                   inDenseSize  = 1,
                         unsigned                   outIndex     = 0,
                         unsigned                   outDenseSize = 1) const
    {
        if (dim == sizesOut[2].size())
        {
            ASSERT_EQ(pOutputBuffer[outIndex], pInputBuffer[inIndex]) << "mismatch at index: " << outIndex;
            return;
        }
        for (unsigned i = 0; i < sizesOut[2][dim]; ++i)
        {
            unsigned inputIndex = (sizesOut[2][dim] == sizesIn[2][dim]) ? i : 0;
            validateResults(pOutputBuffer,
                            pInputBuffer,
                            sizesIn,
                            sizesOut,
                            dim + 1,
                            inIndex + inputIndex * inDenseSize,
                            inDenseSize * sizesIn[2][dim],
                            outIndex + i * outDenseSize,
                            outDenseSize * sizesOut[2][dim]);
            if (HasFailure()) break;
        }
    }
};

// SynGaudiBroadcastTest for tests that are supported on gaudi1 only
class SynGaudiBroadcastTest : public SynTrainingBroadcastTest
{
};

class SynGaudiBroadcastTestGaudi3 : public SynGaudiBroadcastTest
{
};

TEST_P_GC(SynTrainingBroadcastTest, broadcast_test_static_shape)
{
    auto                    params        = GetParam();
    unsigned                broadcastMask = std::get<1>(std::get<0>(params));
    std::array<unsigned, 5> sizes         = {std::get<1>(params),
                                     std::get<2>(params),
                                     std::get<3>(params),
                                     std::get<4>(params),
                                     std::get<5>(params)};
    TestSizeVec              input;
    TestSizeVec              output;
    for (unsigned dim = 0; dim < std::get<0>(std::get<0>(params)); ++dim)
    {
        input.push_back(shouldBeBroadcastDim(broadcastMask, dim) ? 1 : sizes[dim]);
        output.push_back(sizes[dim]);
    }
    run(input, output);
}

INSTANTIATE_TEST_SUITE_P(Sanity,
                         SynTrainingBroadcastTest,
                         // tensor dim, broadcast mask
                         ::testing::Combine(::testing::Values(std::make_tuple<unsigned, unsigned>(1, 0b00001),
                                                              std::make_tuple<unsigned, unsigned>(2, 0b00001),
                                                              std::make_tuple<unsigned, unsigned>(2, 0b00010),
                                                              std::make_tuple<unsigned, unsigned>(2, 0b00011),
                                                              std::make_tuple<unsigned, unsigned>(4, 0b00101),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00001),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00010),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b01010),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10000),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10010),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00110),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10110),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10101),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00101),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b11111)),
                                            ::testing::Values<unsigned>(1, 64, 23),  // dim 0 size
                                            ::testing::Values<unsigned>(48),         // dim 1 size
                                            ::testing::Values<unsigned>(4, 32),      // dim 2 size
                                            ::testing::Values<unsigned>(1, 19, 27),  // dim 3 size
                                            ::testing::Values<unsigned>(10)));       // dim 4 size
TEST_P_GC(SynGaudiBroadcastTestGaudi3, broadcast_test_dynamic_shape)
{
    auto                    params        = GetParam();
    unsigned                broadcastMask = std::get<1>(std::get<0>(params));
    unsigned                dims          = std::get<0>(std::get<0>(params));
    std::array<unsigned, 5> sizes         = {std::get<1>(params),
                                     std::get<2>(params),
                                     std::get<3>(params),
                                     std::get<4>(params),
                                     std::get<5>(params)};

    TestSizeVec              inputMin;
    TestSizeVec              inputMax;
    TestSizeVec              inputActual;

    TestSizeVec outputMin;
    TestSizeVec outputMax;
    TestSizeVec outputActual;
    for (unsigned dim = 0; dim < dims; ++dim)
    {
        unsigned maxSize    = sizes[dim];
        unsigned actualSize = std::max<unsigned>(1, PRIME % maxSize);
        unsigned minSize    = std::max<unsigned>(1, actualSize / 2);
        inputMax.push_back(shouldBeBroadcastDim(broadcastMask, dim) ? 1 : maxSize);
        inputMin.push_back(shouldBeBroadcastDim(broadcastMask, dim) ? 1 : minSize);
        inputActual.push_back(shouldBeBroadcastDim(broadcastMask, dim) ? 1 : actualSize);

        outputMax.push_back(maxSize);
        outputMin.push_back(minSize);
        outputActual.push_back(actualSize);
    }

    std::array<TestSizeVec, 3> sizesIn  = {inputMax, inputMin, inputActual};
    std::array<TestSizeVec, 3> sizesOut = {outputMax, outputMin, outputActual};
    run(sizesIn, sizesOut);
}

INSTANTIATE_TEST_SUITE_P(Sanity,
                         SynGaudiBroadcastTestGaudi3,
                         // tensor dim, broadcast mask
                         ::testing::Combine(::testing::Values(std::make_tuple<unsigned, unsigned>(1, 0b00001),
                                                              std::make_tuple<unsigned, unsigned>(2, 0b00001),
                                                              std::make_tuple<unsigned, unsigned>(2, 0b00010),
                                                              std::make_tuple<unsigned, unsigned>(2, 0b00011),
                                                              std::make_tuple<unsigned, unsigned>(4, 0b00101),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00001),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00010),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b01010),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10000),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10010),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00110),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10110),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b10101),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b00101),
                                                              std::make_tuple<unsigned, unsigned>(5, 0b11111)),
                                            ::testing::Values<unsigned>(1, 64, 23),  // dim 0 size
                                            ::testing::Values<unsigned>(48),         // dim 1 size
                                            ::testing::Values<unsigned>(4, 32),      // dim 2 size
                                            ::testing::Values<unsigned>(1, 19, 27),  // dim 3 size
                                            ::testing::Values<unsigned>(10)));       // dim 4 size

// a lot of test (~3 milions) make run_synapse_test super slow
/*
INSTANTIATE_TEST_SUITE_P(Full,
                         SynGaudiBroadcastTest,
                         // tensor dim, broadcast mask
                         ::testing::Combine(::testing::Combine(::testing::Range<unsigned>(1, 5),
                                                               ::testing::Range<unsigned>(0b00001, 0b11111)),
                                            ::testing::Range<unsigned>(1, 128, 23),       // dim 0 size
                                            ::testing::Range<unsigned>(1, 128, 17),       // dim 1 size
                                            ::testing::Range<unsigned>(1, 128, 32),       // dim 2 size
                                            ::testing::Range<unsigned>(1, 128, 19),       // dim 3 size
                                            ::testing::Range<unsigned>(1, 256, 29)));     // dim 4 size
*/

TEST_F_GC(SynGaudiTwoRunCompareTest, broadcast_test_with_const_input)
{
    ScopedConfigurationChange hbmSizeCfg("HBM_GLOBAL_MEM_SIZE_MEGAS", "256");
    ScopedConfigurationChange constTensorSizeCfg("MAX_CONST_TENSOR_SIZE_BYTES", "0x6400000");

    unsigned broadcastIn1Sizes[] = {4, 261888};
    unsigned broadcastIn1        = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "broadcastIn1",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          broadcastIn1Sizes,
                                          2,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          true,
                                          broadcastIn1Sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned broadcastIn2Sizes[] = {4, 261888, 2};
    unsigned broadcastIn2        = createTensors(1,
                                          INPUT_TENSOR,
                                          false,
                                          "broadcastIn2",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          broadcastIn2Sizes,
                                          3,
                                          syn_type_uint32,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          broadcastIn2Sizes,
                                          synTensorType::SHAPE_TENSOR)[0];

    unsigned broadcastOutSizes[] = {4, 261888, 2};
    unsigned broadcastOut        = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "broadcastOut",
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          broadcastOutSizes,
                                          3,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          broadcastOutSizes,
                                          synTensorType::DATA_TENSOR)[0];

    synNodeId broadcastId;
    addNodeToGraph("broadcast",
                   {broadcastIn1, broadcastIn2},
                   {broadcastOut},
                   nullptr,
                   0,
                   "broadcast",
                   0 /*graphIndex*/,
                   &broadcastId);

    unsigned sliceInSizes[] = {4, 261888, 1};
    unsigned sliceIn        = createTensors(1,
                                     INPUT_TENSOR,
                                     false,
                                     "sliceIn",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     sliceInSizes,
                                     3,
                                     syn_type_uint32,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     sliceInSizes,
                                     synTensorType::SHAPE_TENSOR)[0];

    unsigned       sliceOutSizes[] = {4, 261888, 1};
    unsigned       sliceOut        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "sliceOut",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      sliceOutSizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      sliceOutSizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId      sliceId;
    synSliceParams sliceParams = {.axes   = {0, 1, 2, 0, 0},
                                  .starts = {0, 0, 0, 0, 0},
                                  .ends   = {4, 261888, 1, 0, 0},
                                  .steps  = {1, 1, 1, 0, 0}};
    addNodeToGraph("slice",
                   {broadcastOut, sliceIn},
                   {sliceOut},
                   &sliceParams,
                   sizeof(sliceParams),
                   "slice",
                   0 /*graphIndex*/,
                   &sliceId);

    unsigned reshapeInSizes[] = {4, 261888};
    unsigned reshapeIn        = createTensors(1,
                                       INPUT_TENSOR,
                                       false,
                                       "reshapeIn",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       reshapeInSizes,
                                       2,
                                       syn_type_uint32,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       reshapeInSizes,
                                       synTensorType::SHAPE_TENSOR)[0];

    unsigned reshapeOutSizes[] = {4, 261888};
    unsigned reshapeOut        = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false,
                                        "reshapeOut",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        reshapeOutSizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false,
                                        reshapeOutSizes,
                                        synTensorType::DATA_TENSOR)[0];

    synNodeId reshapeId;
    addNodeToGraph("reshape", {sliceOut, reshapeIn}, {reshapeOut}, nullptr, 0, "reshape", 0 /*graphIndex*/, &reshapeId);

    unsigned gatherInSizes[] = {6000};
    unsigned gatherIn        = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "gatherIn",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      gatherInSizes,
                                      1,
                                      syn_type_int32,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      gatherInSizes,
                                      synTensorType::DATA_TENSOR)[0];

    unsigned gatherOutSizes[] = {4, 6000};
    unsigned gatherOut        = createTensors(1,
                                       OUTPUT_TENSOR,
                                       true,
                                       "gatherOut",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       gatherOutSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       gatherOutSizes,
                                       synTensorType::DATA_TENSOR)[0];

    synNodeId               gatherId;
    ns_GatherKernel::Params gatherParams = {0};
    gatherParams.axis                    = 1;
    addNodeToGraph("gather_fwd_f32",
                   {reshapeOut, gatherIn},
                   {gatherOut},
                   &gatherParams,
                   sizeof(ns_GatherKernel::Params),
                   "gather",
                   0 /*graphIndex*/,
                   &gatherId);

    addConfigurationToRun(FIRST_RUN, "MAKE_BROADCAST_PHYSICAL", "false");
    addConfigurationToRun(SECOND_RUN, "MAKE_BROADCAST_PHYSICAL", "true");

    compareRunsResults({gatherOut});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, tile_test_with_const_input)
{
    ScopedConfigurationChange hbmSizeCfg("HBM_GLOBAL_MEM_SIZE_MEGAS", "256");
    ScopedConfigurationChange constTensorSizeCfg("MAX_CONST_TENSOR_SIZE_BYTES", "0x6400000");

    unsigned tileInSizes[] = {1, 1, 1, 1};
    unsigned tileIn        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "tileIn",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    tileInSizes,
                                    4,
                                    syn_type_single,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    true,
                                    tileInSizes,
                                    synTensorType::DATA_TENSOR)[0];

    unsigned tileOutSizes[] = {512, 256, 1, 1};
    unsigned tileOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     true,
                                     "tileOut",
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     tileOutSizes,
                                     4,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     tileOutSizes,
                                     synTensorType::DATA_TENSOR)[0];

    unsigned char tileParams[] = {0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("tile_fwd_f32", {tileIn}, {tileOut}, (void*)tileParams, 20, "TILE");

    addConfigurationToRun(FIRST_RUN, "MAKE_BROADCAST_PHYSICAL", "false");
    addConfigurationToRun(SECOND_RUN, "MAKE_BROADCAST_PHYSICAL", "true");

    compareRunsResults({tileOut});
}
