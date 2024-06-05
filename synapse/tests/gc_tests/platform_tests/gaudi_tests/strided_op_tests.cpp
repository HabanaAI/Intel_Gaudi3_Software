#include "gc_dynamic_shapes_infra.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "recipe.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "tensor.h"
#include "types.h"
#include "h2d_tensors.h"

#include "syn_singleton.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include "gtest/gtest.h"
#include "gtest/gtest-param-test.h"

static unsigned countNumRealNodes(const synGraphHandle& handle)
{
    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(handle);
    unsigned           ret   = 0;
    for (const NodePtr& n : graph->getNodes())
    {
        if (n && !n->isLogicalOperation())
        {
            ret++;
        }
    }
    return ret;
}

static unsigned countNumViewInsertNodes(const synGraphHandle& handle)
{
    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(handle);
    unsigned           ret   = 0;
    for (const NodePtr& n : graph->getNodes())
    {
        if (n && (n->getNodeType() == Node::TYPE_STRIDED_VIEW || n->getNodeType() == Node::TYPE_STRIDED_INSERT))
        {
            ret++;
        }
    }
    return ret;
}

class SynTrainingStridedViewInsertTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<bool /* stridedInsert or stridedView */,
                                                std::tuple<TestSizes /* inputSize */,
                                                           TestSizes /* outputSize */,
                                                           StrideArray /* strides */,
                                                           uint64_t /* base offset */>>>
{
    static const unsigned NUM_DIMS = 4;

protected:
    void
    runStridedView(const TestSizes& realSize, const TestSizes& viewSize, const StrideArray& strides, uint64_t offset)
    {
        synStridedOpParams params;
        memset(&params, 0, sizeof(params));
        params.baseOffset = offset;
        for (int i = 0; i < NUM_DIMS; i++)
        {
            params.strides[i] = strides[i];
        }

        unsigned sizesIn[NUM_DIMS]  = {0};
        unsigned sizesOut[NUM_DIMS] = {0};
        memcpy(sizesIn, realSize.data(), sizeof(sizesIn));
        memcpy(sizesOut, viewSize.data(), sizeof(sizesOut));

        unsigned in  = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          sizesIn,
                                          NUM_DIMS,
                                          syn_type_single,
                                          nullptr,
                                          "input");
        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizesOut,
                                           NUM_DIMS,
                                           syn_type_single,
                                           nullptr,
                                           "out");
        addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {in}, {out}, &params, sizeof(params), "strided_view");

        compileTopology();
        runTopology();

        float* outputData = (float*)m_hostBuffers[out];
        float* inputData  = (float*)m_hostBuffers[in];

        unsigned outIndex = 0;
        for (unsigned i = 0; i < sizesOut[3]; i++)
            for (unsigned j = 0; j < sizesOut[2]; j++)
                for (unsigned k = 0; k < sizesOut[1]; k++)
                    for (unsigned l = 0; l < sizesOut[0]; l++)
                    {
                        unsigned inIndex = i * params.strides[3] + j * params.strides[2] + k * params.strides[1] +
                                           l * params.strides[0] + params.baseOffset;
                        ASSERT_EQ(inputData[inIndex], outputData[outIndex])
                            << "Mismatch at index " << i << "," << j << "," << k << "," << l
                            << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
                        outIndex++;
                    }
    }

    void
    runStridedInsert(const TestSizes& realSize, const TestSizes& viewSize, const StrideArray& strides, uint64_t offset)
    {
        synStridedOpParams params;
        memset(&params, 0, sizeof(params));
        params.baseOffset = offset;
        for (int i = 0; i < NUM_DIMS; i++)
        {
            params.strides[i] = strides[i];
        }

        unsigned sizesReal[NUM_DIMS] = {0};
        unsigned sizesView[NUM_DIMS] = {0};
        memcpy(sizesReal, realSize.data(), sizeof(sizesReal));
        memcpy(sizesView, viewSize.data(), sizeof(sizesView));

        unsigned inReal = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              sizesReal,
                                              NUM_DIMS,
                                              syn_type_single,
                                              nullptr,
                                              "input_real");

        unsigned inView = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              sizesView,
                                              NUM_DIMS,
                                              syn_type_single,
                                              nullptr,
                                              "input_view");

        unsigned viewTensor =
            createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesView, NUM_DIMS, syn_type_single);

        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizesReal,
                                           NUM_DIMS,
                                           syn_type_single,
                                           nullptr,
                                           "out");

        addNodeToGraph("relu_fwd_f32", {inView}, {viewTensor}, nullptr, 0, "relu");
        addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                       {inReal, viewTensor},
                       {out},
                       &params,
                       sizeof(params),
                       "strided_insert");

        compileTopology();

        runTopology();

        float* outputData    = (float*)m_hostBuffers[out];
        float* inputViewData = (float*)m_hostBuffers[inView];
        float* inputData     = (float*)m_hostBuffers[inReal];

        // count number of diffs between the original tensor and the output tensor
        unsigned numDiffs = 0;
        for (unsigned i = 0; i < sizesReal[0] * sizesReal[1] * sizesReal[2] * sizesReal[3]; i++)
        {
            if (outputData[i] != inputData[i])
            {
                numDiffs++;
            }
        }
        ASSERT_LE(numDiffs, sizesView[0] * sizesView[1] * sizesView[2] * sizesView[3])
            << "too many diffs in original tensor!";

        unsigned viewIndex = 0;
        for (unsigned i = 0; i < sizesView[3]; i++)
            for (unsigned j = 0; j < sizesView[2]; j++)
                for (unsigned k = 0; k < sizesView[1]; k++)
                    for (unsigned l = 0; l < sizesView[0]; l++)
                    {
                        // offset for view index in the output
                        unsigned realOutIndex = i * params.strides[3] + j * params.strides[2] + k * params.strides[1] +
                                                l * params.strides[0] + params.baseOffset;

                        // compare inserted data
                        float viewValue = inputViewData[viewIndex];
                        viewValue       = (viewValue < 0) ? 0 : viewValue;
                        ASSERT_EQ(viewValue, outputData[realOutIndex])
                            << "Mismatch at index " << i << "," << j << "," << k << "," << l
                            << " Expected: " << viewValue << " Result: " << outputData[realOutIndex];
                        viewIndex++;
                    }
    }

    void runSingleTest()
    {
        auto        isView = std::get<0>(GetParam());
        const auto& params = std::get<1>(GetParam());
        if (isView)
        {
            runStridedView(std::get<0>(params), std::get<1>(params), std::get<2>(params), std::get<3>(params));
        }
        else
        {
            runStridedInsert(std::get<0>(params), std::get<1>(params), std::get<2>(params), std::get<3>(params));
        }
    }
};

TEST_P_GC(SynTrainingStridedViewInsertTest, viewInsertTest)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SynTrainingStridedViewInsertTest,
    ::testing::Combine(
        ::testing::ValuesIn({true, false}),
        // input sizes,        ouput sizes      ,   strides (elements), offset
        ::testing::Values(
            std::make_tuple(TestSizes {2, 4, 1, 1}, TestSizes {2, 2, 1, 1}, StrideArray {1, 4, 4, 4}, 0),
            std::make_tuple(TestSizes {2, 4, 1, 1}, TestSizes {2, 2, 1, 1}, StrideArray {1, 6, 6, 6}, 0),
            std::make_tuple(TestSizes {4, 2, 3, 4}, TestSizes {4, 3, 2, 2}, StrideArray {1, 8, 4, 48}, 0),
            std::make_tuple(TestSizes {2, 4, 3, 1}, TestSizes {2, 2, 3, 1}, StrideArray {1, 4, 8, 8}, 2),
            std::make_tuple(TestSizes {4, 2, 1, 1}, TestSizes {2, 2, 1, 1}, StrideArray {2, 4, 4, 4}, 1))));

/*
        +-------+     +-------+     +-------+     +-------+     +-------+     +-------+     +-------+
  t1    |       | t2  |       | t3  |       | t4  |       | t5  |       | t6  |       | t7  |       |  t8
+------>+ ReLU1 +--+->+ View1 +---->+  Add1 +---->+Insert1+--+->+ View2 +---->+ Add2  +---->+Insert1+------->
        |       |  |  |       |     |       |     |       |  |  |       |     |       |     |       |
        +-------+  |  +-------+     +-------+     +---+---+  |  +-------+     +-------+     +---+---+
                   |                                  ^      |                                  ^
                   |                                  |      |                                  |
                   +----------------------------------+      +----------------------------------+

*/
TEST_F_GC(SynTrainingTestInfra, stridedInsertView)
{
    static const unsigned NUM_DIMS             = 3;
    unsigned              inSizes[NUM_DIMS]    = {4, 3, 2};
    unsigned              sliceSize1[NUM_DIMS] = {4, 2, 2};
    unsigned              sliceSize2[NUM_DIMS] = {4, 1, 2};

    unsigned tAdd1 = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sliceSize1,
                                         NUM_DIMS,
                                         syn_type_single,
                                         nullptr,
                                         "tAdd1");
    unsigned tAdd2 = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sliceSize2,
                                         NUM_DIMS,
                                         syn_type_single,
                                         nullptr,
                                         "tAdd2");
    unsigned t1    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "in");
    unsigned t2    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inSizes, NUM_DIMS, syn_type_single);
    unsigned t3    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize1, NUM_DIMS, syn_type_single);
    unsigned t4    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize1, NUM_DIMS, syn_type_single);
    unsigned t5    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inSizes, NUM_DIMS, syn_type_single);
    unsigned t6    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize2, NUM_DIMS, syn_type_single);
    unsigned t7    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize2, NUM_DIMS, syn_type_single);
    unsigned t8    = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "out");

    synStridedOpParams view1Params = {0};
    view1Params.strides[0]         = 1;
    view1Params.strides[1]         = view1Params.strides[0] * inSizes[0];
    view1Params.strides[2]         = view1Params.strides[1] * inSizes[1];

    synStridedOpParams view2Params = view1Params;
    view2Params.baseOffset         = 4;

    addNodeToGraph("relu_fwd_f32", {t1}, {t2}, nullptr, 0, "relu1");
    addNodeToGraph("strided_view", {t2}, {t3}, &view1Params, sizeof(view1Params), "view1");
    addNodeToGraph("add_fwd_f32", {t3, tAdd1}, {t4}, nullptr, 0, "add1");
    addNodeToGraph("strided_insert", {t2, t4}, {t5}, &view1Params, sizeof(view1Params), "insert1");
    addNodeToGraph("strided_view", {t5}, {t6}, &view2Params, sizeof(view1Params), "view2");
    addNodeToGraph("add_fwd_f32", {t6, tAdd2}, {t7}, nullptr, 0, "add2");
    addNodeToGraph("strided_insert", {t5, t7}, {t8}, &view2Params, sizeof(view2Params), "insert2");

    compileTopology();

    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 3);  // verify no internal memcopy nodes were added

    runTopology();

    float* inputData = (float*)m_hostBuffers[t1];
    float* add1Data  = (float*)m_hostBuffers[tAdd1];
    float* add2Data  = (float*)m_hostBuffers[tAdd2];

    // calculate expected output
    unsigned numElements = inSizes[0] * inSizes[1] * inSizes[2];
    float    expected[numElements];
    memcpy(expected, inputData, inSizes[0] * inSizes[1] * inSizes[2] * sizeof(float));

    // relu
    for (unsigned i = 0; i < numElements; i++)
    {
        expected[i] = expected[i] > 0 ? expected[i] : 0;
    }

    // add1
    unsigned idx = 0;
    for (unsigned h = 0; h < sliceSize1[2]; h++)
    {
        for (unsigned w = 0; w < sliceSize1[1]; w++)
        {
            for (unsigned c = 0; c < sliceSize1[0]; c++)
            {
                unsigned offset = view1Params.baseOffset;
                offset += view1Params.strides[0] * c;
                offset += view1Params.strides[1] * w;
                offset += view1Params.strides[2] * h;
                expected[offset] += add1Data[idx++];
            }
        }
    }

    // add2
    idx = 0;
    for (unsigned h = 0; h < sliceSize2[2]; h++)
    {
        for (unsigned w = 0; w < sliceSize2[1]; w++)
        {
            for (unsigned c = 0; c < sliceSize2[0]; c++)
            {
                unsigned offset = view2Params.baseOffset;
                offset += view2Params.strides[0] * c;
                offset += view2Params.strides[1] * w;
                offset += view2Params.strides[2] * h;
                expected[offset] += add2Data[idx++];
            }
        }
    }

    float* outputData = (float*)m_hostBuffers[t8];
    // compare results
    for (unsigned i = 0; i < numElements; i++)
    {
        ASSERT_EQ(expected[i], outputData[i])
            << "Mismatch at index " << i << " Expected: " << expected[i] << " Result: " << outputData[i];
    }
}

/*
        +-------+     +-------+     +-------+     +-------+
  t1    |       | t2  |       | t3  |       | t4  |       | t5
+------>+ ReLU1 +--+->+ View1 +---->+  Add1 +---->+Insert1+------->
        |       |  |  |       |     |       |     |       |
        +-------+  |  +-------+     +-------+     +---+---+
                   |                                  ^
                   |                                  |
                   +----------------------------------+
Add1 uses the same memory for input and output (different strides).
A memcopy should be planted, otherwise we will get incorrect results.
*/
TEST_F_GC(SynTrainingTestInfra, stridedOvelapTest)
{
    static const unsigned NUM_DIMS             = 2;
    unsigned              inSizes[NUM_DIMS]    = {64, 256};
    unsigned              sliceSize1[NUM_DIMS] = {64, 128};

    unsigned tAdd1 = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sliceSize1,
                                         NUM_DIMS,
                                         syn_type_single,
                                         nullptr,
                                         "tAdd1");
    unsigned t1    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "in");
    unsigned t2    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inSizes, NUM_DIMS, syn_type_single);
    unsigned t3    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize1, NUM_DIMS, syn_type_single);
    unsigned t4    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize1, NUM_DIMS, syn_type_single);
    unsigned t5    = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "out");

    synStridedOpParams view1Params = {0};
    view1Params.strides[0]         = 1;
    view1Params.strides[1]         = 64;
    view1Params.baseOffset         = 0;

    synStridedOpParams view2Params = view1Params;
    view2Params.strides[0]         = 1;
    view2Params.strides[1]         = 128;
    view2Params.baseOffset         = 0;

    addNodeToGraph("relu_fwd_f32", {t1}, {t2}, nullptr, 0, "relu1");
    addNodeToGraph("strided_view", {t2}, {t3}, &view1Params, sizeof(view1Params), "view1");
    addNodeToGraph("add_fwd_f32", {t3, tAdd1}, {t4}, nullptr, 0, "add1");
    addNodeToGraph("strided_insert", {t2, t4}, {t5}, &view2Params, sizeof(view2Params), "insert1");

    compileTopology();

    runTopology();

    float* inputData = (float*)m_hostBuffers[t1];
    float* add1Data  = (float*)m_hostBuffers[tAdd1];

    // calculate expected output
    unsigned numElements = inSizes[0] * inSizes[1];
    float    expected[numElements];
    float    addOut[sliceSize1[0] * sliceSize1[1]];
    memcpy(expected, inputData, inSizes[0] * inSizes[1] * sizeof(float));

    // relu
    for (unsigned i = 0; i < numElements; i++)
    {
        expected[i] = expected[i] > 0 ? expected[i] : 0;
    }

    // add1
    unsigned idx = 0;
    for (unsigned w = 0; w < sliceSize1[1]; w++)
    {
        for (unsigned c = 0; c < sliceSize1[0]; c++)
        {
            unsigned offsetSrc = view1Params.baseOffset;
            offsetSrc += view1Params.strides[0] * c + view1Params.strides[1] * w;
            addOut[idx] = expected[offsetSrc] + add1Data[idx];
            idx++;
        }
    }

    // insert
    idx = 0;
    for (unsigned w = 0; w < sliceSize1[1]; w++)
    {
        for (unsigned c = 0; c < sliceSize1[0]; c++)
        {
            unsigned offsetDst = view2Params.baseOffset;
            offsetDst += view2Params.strides[0] * c + view2Params.strides[1] * w;
            expected[offsetDst] = addOut[idx];
            idx++;
        }
    }

    float* outputData = (float*)m_hostBuffers[t5];
    // compare results
    for (unsigned i = 0; i < numElements; i++)
    {
        ASSERT_EQ(expected[i], outputData[i])
            << "Mismatch at index " << i << " Expected: " << expected[i] << " Result: " << outputData[i];
    }
}

class SynGaudiStridedViewDynamicTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_view_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 3;

    unsigned inSizesMax[NUM_DIMS]      = {4, 32, 64};
    unsigned inSizesMin[NUM_DIMS]      = {4, 16, 32};
    unsigned outSizesMax[NUM_DIMS - 1] = {4, 32 * 16};
    unsigned outSizesMin[NUM_DIMS - 1] = {4, 16 * 16};

    synStridedOpParams params;
    memset(&params, 0, sizeof(params));
    params.baseOffset = 2;
    params.strides[0] = 1;
    params.strides[1] = 4;

    unsigned in    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizesMax,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input",
                                      0,
                                      0,
                                      nullptr,
                                      inSizesMin);
    unsigned shape = createShapeTensor(INPUT_TENSOR, outSizesMax, outSizesMin, NUM_DIMS - 1);
    unsigned out   = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       outSizesMin);
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {in, shape}, {out}, &params, sizeof(params), "strided_view");

    compileTopology();

    unsigned inActualSizes[]  = {4, 22, 32};
    unsigned outActualSizes[] = {4, 18 * 16};
    setActualSizes(in, inActualSizes);
    setActualSizes(out, outActualSizes);
    setActualSizes(shape, outActualSizes);

    runTopology();

    float* outputData = (float*)m_hostBuffers[out];
    float* inputData  = (float*)m_hostBuffers[in];

    unsigned outIndex = 0;
    for (unsigned j = 0; j < outActualSizes[1]; j++)
        for (unsigned l = 0; l < outActualSizes[0]; l++)
        {
            unsigned inIndex = j * params.strides[1] + l * params.strides[0] + params.baseOffset;
            ASSERT_EQ(inputData[inIndex], outputData[outIndex])
                << "Mismatch at index " << l << " Expected: " << inputData[inIndex]
                << " Result: " << outputData[outIndex];
            outIndex++;
        }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_insert_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 3;

    unsigned originalSizesMax[NUM_DIMS]   = {4, 32, 64};
    unsigned originalSizesMin[NUM_DIMS]   = {4, 16, 32};
    unsigned insertSizesMax[NUM_DIMS - 1] = {4, 32 * 16};
    unsigned insertSizesMin[NUM_DIMS - 1] = {4, 16 * 16};

    synStridedOpParams params;
    memset(&params, 0, sizeof(params));
    params.baseOffset = 2;
    params.strides[0] = 1;
    params.strides[1] = 4;

    unsigned inReal = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          originalSizesMax,
                                          NUM_DIMS,
                                          syn_type_single,
                                          nullptr,
                                          "input_real",
                                          0,
                                          0,
                                          nullptr,
                                          originalSizesMin);

    unsigned inView = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          insertSizesMax,
                                          NUM_DIMS - 1,
                                          syn_type_single,
                                          nullptr,
                                          "input_view",
                                          0,
                                          0,
                                          nullptr,
                                          insertSizesMin);

    unsigned viewTensor = createTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       insertSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       insertSizesMin);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       originalSizesMax,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       originalSizesMin);

    addNodeToGraph("relu_fwd_f32", {inView}, {viewTensor}, nullptr, 0, "relu");
    addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                   {inReal, viewTensor},
                   {out},
                   &params,
                   sizeof(params),
                   "strided_insert");

    compileTopology();

    unsigned originalActualSizes[] = {4, 22, 32};
    unsigned insertActualSizes[]   = {4, 18 * 16};
    setActualSizes(inReal, originalActualSizes);
    setActualSizes(inView, insertActualSizes);
    setActualSizes(out, originalActualSizes);

    runTopology();

    float* outputData    = (float*)m_hostBuffers[out];
    float* inputViewData = (float*)m_hostBuffers[inView];
    float* inputData     = (float*)m_hostBuffers[inReal];

    // count number of diffs between the original tensor and the output tensor
    unsigned numDiffs = 0;
    for (unsigned i = 0; i < originalActualSizes[0] * originalActualSizes[1] * originalActualSizes[2]; i++)
    {
        if (outputData[i] != inputData[i])
        {
            numDiffs++;
        }
    }
    ASSERT_LE(numDiffs, insertActualSizes[0] * insertActualSizes[1]) << "too many diffs in original tensor!";

    unsigned viewIndex = 0;
    for (unsigned k = 0; k < insertActualSizes[1]; k++)
        for (unsigned l = 0; l < insertActualSizes[0]; l++)
        {
            // offset for view index in the output
            unsigned realOutIndex = k * params.strides[1] + l * params.strides[0] + params.baseOffset;

            // compare inserted data
            float viewValue = inputViewData[viewIndex];
            viewValue       = (viewValue < 0) ? 0 : viewValue;
            ASSERT_EQ(viewValue, outputData[realOutIndex])
                << "Mismatch at index " << k << "," << l << " Expected: " << viewValue
                << " Result: " << outputData[realOutIndex];
            viewIndex++;
        }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_view_dynamic_fcd, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned inSizesMax[NUM_DIMS]      = {8, 8};
    unsigned inSizesMin[NUM_DIMS]      = {4, 4};
    unsigned outSizesMax[NUM_DIMS - 1] = {(8 * 8) / 2};
    unsigned outSizesMin[NUM_DIMS - 1] = {(4 * 4) / 2};

    synStridedOpParams params;
    memset(&params, 0, sizeof(params));
    params.baseOffset = 0;
    params.strides[0] = 2;

    unsigned in    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizesMax,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input",
                                      0,
                                      0,
                                      nullptr,
                                      inSizesMin);
    unsigned shape = createShapeTensor(INPUT_TENSOR, outSizesMax, outSizesMin, NUM_DIMS - 1);
    unsigned out   = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       outSizesMin);
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {in, shape}, {out}, &params, sizeof(params), "strided_view");

    compileTopology();

    unsigned inActualSizes[]  = {6, 6};
    unsigned outActualSizes[] = {(6 * 6) / 2};
    setActualSizes(in, inActualSizes);
    setActualSizes(out, outActualSizes);
    setActualSizes(shape, outActualSizes);

    runTopology();

    float* outputData = (float*)m_hostBuffers[out];
    float* inputData  = (float*)m_hostBuffers[in];

    unsigned outIndex = 0;
    for (unsigned l = 0; l < outActualSizes[0]; l++)
    {
        unsigned inIndex = l * params.strides[0] + params.baseOffset;
        ASSERT_EQ(inputData[inIndex], outputData[outIndex])
            << "Mismatch at index " << l << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
        outIndex++;
    }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_insert_dynamic_fcd, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned originalSizesMax[NUM_DIMS]   = {8, 8};
    unsigned originalSizesMin[NUM_DIMS]   = {4, 4};
    unsigned insertSizesMax[NUM_DIMS - 1] = {(8 * 8) / 2};
    unsigned insertSizesMin[NUM_DIMS - 1] = {(4 * 4) / 2};

    synStridedOpParams params;
    memset(&params, 0, sizeof(params));
    params.baseOffset = 0;
    params.strides[0] = 2;

    unsigned inReal = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          originalSizesMax,
                                          NUM_DIMS,
                                          syn_type_single,
                                          nullptr,
                                          "input_real",
                                          0,
                                          0,
                                          nullptr,
                                          originalSizesMin);

    unsigned inView = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          insertSizesMax,
                                          NUM_DIMS - 1,
                                          syn_type_single,
                                          nullptr,
                                          "input_view",
                                          0,
                                          0,
                                          nullptr,
                                          insertSizesMin);

    unsigned viewTensor = createTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       insertSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       insertSizesMin);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       originalSizesMax,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       originalSizesMin);

    addNodeToGraph("relu_fwd_f32", {inView}, {viewTensor}, nullptr, 0, "relu");
    addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                   {inReal, viewTensor},
                   {out},
                   &params,
                   sizeof(params),
                   "strided_insert");

    compileTopology();

    unsigned originalActualSizes[] = {6, 6};
    unsigned insertActualSizes[]   = {(6 * 6) / 2};
    setActualSizes(inReal, originalActualSizes);
    setActualSizes(inView, insertActualSizes);
    setActualSizes(out, originalActualSizes);

    runTopology();

    float* outputData    = (float*)m_hostBuffers[out];
    float* inputViewData = (float*)m_hostBuffers[inView];
    float* inputData     = (float*)m_hostBuffers[inReal];

    for (unsigned i = 0; i < insertActualSizes[0]; i++)
    {
        float expected = (i % 2 == 0) ? std::max(inputViewData[i / 2], 0.0f) : inputData[i];
        ASSERT_EQ(expected, outputData[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << outputData[i];
    }
}

TEST_F_GC(SynTrainingTestInfra, strided_view_high_rank)
{
    static const unsigned NUM_DIMS = 5;
    synStridedOpParams    params;
    memset(&params, 0, sizeof(params));
    params.baseOffset = 0;
    params.strides[0] = 32;
    params.strides[1] = 1;
    params.strides[2] = 32;
    params.strides[3] = 512;
    params.strides[4] = 1024;

    unsigned sizesIn[NUM_DIMS]  = {32, 1, 16, 2, 4};
    unsigned sizesOut[NUM_DIMS] = {1, 32, 16, 2, 4};

    unsigned in  = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sizesIn,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input");
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       sizesOut,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out");
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {in}, {out}, &params, sizeof(params), "strided_view");

    compileTopology();
    runTopology();

    float* outputData = (float*)m_hostBuffers[out];
    float* inputData  = (float*)m_hostBuffers[in];

    unsigned outIndex = 0;
    for (unsigned n = 0; n < sizesOut[4]; n++)
        for (unsigned i = 0; i < sizesOut[3]; i++)
            for (unsigned j = 0; j < sizesOut[2]; j++)
                for (unsigned k = 0; k < sizesOut[1]; k++)
                    for (unsigned l = 0; l < sizesOut[0]; l++)
                    {
                        unsigned inIndex = n * params.strides[4] + i * params.strides[3] + j * params.strides[2] +
                                           k * params.strides[1] + l * params.strides[0] + params.baseOffset;
                        ASSERT_EQ(inputData[inIndex], outputData[outIndex])
                            << "Mismatch at index " << i << "," << j << "," << k << "," << l
                            << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
                        outIndex++;
                    }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_view_dynamic_fcd_dynamic_param, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned inSizesMax[NUM_DIMS]      = {8, 8};
    unsigned inSizesMin[NUM_DIMS]      = {4, 4};
    unsigned outSizesMax[NUM_DIMS - 1] = {(8 * 8) / 2};
    unsigned outSizesMin[NUM_DIMS - 1] = {(4 * 4) / 2};

    unsigned stridesSize[NUM_DIMS - 1] = {2};
    unsigned offsetSize[1]             = {0};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizesMax,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input",
                                      0,
                                      0,
                                      nullptr,
                                      inSizesMin);

    unsigned shape  = createShapeTensor(INPUT_TENSOR, outSizesMax, outSizesMin, NUM_DIMS - 1);
    unsigned stride = createShapeTensor(INPUT_TENSOR, stridesSize, stridesSize, NUM_DIMS - 1);
    unsigned offset = createShapeTensor(INPUT_TENSOR, offsetSize, offsetSize, 1);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       outSizesMin);
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName,
                   {in, shape, stride, offset},
                   {out},
                   nullptr,
                   0,
                   "strided_view");

    compileTopology();

    unsigned inActualSizes[]  = {6, 6};
    unsigned outActualSizes[] = {(6 * 6) / 2};
    setActualSizes(in, inActualSizes);
    setActualSizes(out, outActualSizes);
    setActualSizes(shape, outActualSizes);

    runTopology();

    float* outputData = (float*)m_hostBuffers[out];
    float* inputData  = (float*)m_hostBuffers[in];

    unsigned outIndex = 0;
    for (unsigned l = 0; l < outActualSizes[0]; l++)
    {
        unsigned inIndex = l * stridesSize[0] + offsetSize[0];
        ASSERT_EQ(inputData[inIndex], outputData[outIndex])
            << "Mismatch at index " << l << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
        outIndex++;
    }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest,
          strided_view_dynamic_fcd_dynamic_param_with_h2d,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned inSizesMax[NUM_DIMS]      = {8, 8};
    unsigned inSizesMin[NUM_DIMS]      = {4, 4};
    unsigned outSizesMax[NUM_DIMS - 1] = {(8 * 8) / 2};
    unsigned outSizesMin[NUM_DIMS - 1] = {(4 * 4) / 2};

    synDynamicStridedDmaH2dTensor stridesData[2] = {{1, 0, {2, 1, 1, 1, 1}}, {1, 0, {2, 1, 1, 1, 1}}};
    unsigned                         stridesSize[1] = {sizeof(synDynamicStridedDmaH2dTensor) / sizeof(unsigned)};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizesMax,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input",
                                      0,
                                      0,
                                      nullptr,
                                      inSizesMin);

    unsigned shape  = createShapeTensor(INPUT_TENSOR, outSizesMax, outSizesMin, NUM_DIMS - 1, syn_type_uint32, "shape");
    unsigned stride = createHost2DeviceTensor(INPUT_TENSOR, stridesSize, (unsigned*)stridesData, 1, "strides");

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       outSizesMin);
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {in, shape, stride}, {out}, nullptr, 0, "strided_view");

    compileTopology();

    unsigned inActualSizes[]  = {6, 6};
    unsigned outActualSizes[] = {(6 * 6) / 2};
    setActualSizes(in, inActualSizes);
    setActualSizes(out, outActualSizes);
    setActualSizes(shape, outActualSizes);
    setActualScalarParametersData(stride, stridesData, sizeof(synDynamicStridedDmaH2dTensor));

    runTopology();

    float* outputData = (float*)m_hostBuffers[out];
    float* inputData  = (float*)m_hostBuffers[in];

    unsigned outIndex = 0;
    for (unsigned l = 0; l < outActualSizes[0]; l++)
    {
        unsigned inIndex = l * stridesData[0].strides[0] + stridesData[0].offset;
        ASSERT_EQ(inputData[inIndex], outputData[outIndex])
            << "Mismatch at index " << l << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
        outIndex++;
    }
}

/*
This test is covering the scenario of multiple IH2D launches for the same recipe in parallel.
We first create 10 sets of tensors for the launches while we create a recipe only for the first set.
Second, we copy all tensors to HBM so we can run synLaunch fast on the device for each set and test parallelism.
Each launch is using a different set of tensors with different sizes but with the same recipe, this is why we use the same tensors ID's.
Finally, after synchronizing the compute stream, we verify all result are correct for each launch.
*/

TEST_F_GC(SynGaudiStridedViewDynamicTest,
          strided_view_dynamic_fcd_dynamic_param_with_h2d_multiple_launches,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned        NUM_OF_LAUNCHES = 10;
    static const unsigned NUM_DIMS = 2;
    synStatus             status;
    unsigned              graphIndex = 0;

    unsigned inSizesMax[NUM_DIMS]      = {22, 22};
    unsigned inSizesMin[NUM_DIMS]      = {4, 4};
    unsigned outSizesMax[NUM_DIMS - 1] = {(22 * 22) / 2};
    unsigned outSizesMin[NUM_DIMS - 1] = {(4 * 4) / 2};

    unsigned inActualSizes[NUM_OF_LAUNCHES][2]  = {{4, 4}, {6, 6}, {8, 8}, {10, 10}, {12, 12}, {14, 14}, {16, 16}, {18, 18}, {20, 20}, {22, 22}};
    unsigned outActualSizes[NUM_OF_LAUNCHES][2] = {{(4 * 4) / 2}, {(6 * 6) / 2}, {(8 * 8) / 2}, {(10 * 10) / 2}, {(12 * 12) / 2}, {(14 * 14) / 2},
                                                   {(16 * 16) / 2}, {(18 * 18) / 2}, {(20 * 20) / 2}, {(22 * 22) / 2}};

    synDynamicStridedDmaH2dTensor stridesData[2] = {{1, 0, {2, 1, 1, 1, 1}}, {1, 0, {2, 1, 1, 1, 1}}};
    unsigned                         stridesSize[1] = {sizeof(synDynamicStridedDmaH2dTensor) / sizeof(unsigned)};

    unsigned in[NUM_OF_LAUNCHES];
    unsigned shape[NUM_OF_LAUNCHES];
    unsigned stride[NUM_OF_LAUNCHES];
    unsigned out[NUM_OF_LAUNCHES];

    // Create NUM_OF_LAUNCHES tensors for multi launch test
    for (unsigned i = 0; i < NUM_OF_LAUNCHES; i++)
    {
        std::string inputPersistTensorName  = "input_" + std::to_string(i);
        std::string shapeTensorName         = "shape_" + std::to_string(i);
        std::string host2DeviceTensorName   = "strides_" + std::to_string(i);
        std::string outputPersistTensorName = "out_" + std::to_string(i);

        in[i] = createPersistTensor(INPUT_TENSOR,
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    inSizesMax,
                                    NUM_DIMS,
                                    syn_type_single,
                                    nullptr,
                                    inputPersistTensorName.c_str(),
                                    0,
                                    0,
                                    nullptr,
                                    inSizesMin);

        shape[i]  = createShapeTensor(INPUT_TENSOR, outSizesMax, outSizesMin, NUM_DIMS - 1, syn_type_uint32, shapeTensorName.c_str(), 0);

        stride[i] = createHost2DeviceTensor(INPUT_TENSOR, stridesSize, (unsigned*)stridesData, 1, host2DeviceTensorName.c_str(), 0);

        out[i] = createPersistTensor(OUTPUT_TENSOR,
                                     MEM_INIT_ALL_ZERO,
                                     nullptr,
                                     outSizesMax,
                                     NUM_DIMS - 1,
                                     syn_type_single,
                                     nullptr,
                                     outputPersistTensorName.c_str(),
                                     0,
                                     0,
                                     nullptr,
                                     outSizesMin);
    }
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {in[0], shape[0], stride[0]}, {out[0]}, nullptr, 0, "strided_view");

    compileTopology();

    for (unsigned i = 0; i < NUM_OF_LAUNCHES; i++)
    {
        setActualSizes(in[i], inActualSizes[i]);
        setActualSizes(out[i], outActualSizes[i]);
        setActualSizes(shape[i], outActualSizes[i]);
        setActualScalarParametersData(stride[i], stridesData, sizeof(synDynamicStridedDmaH2dTensor));
    }

    bool        initPersistentOutputs   = false;
    GraphData&  graphData               = getGraph(graphIndex);
    uint64_t    programAddress          = 0;
    std::vector<synLaunchTensorInfoExt> concatTensors;

    // Copy all tensors to HBM - returns the persistent tensors for the launch
    copyInputTensors(graphIndex, programAddress, concatTensors, initPersistentOutputs);

    uint64_t inputTensorId  = concatTensors[0].tensorId;
    uint64_t h2dTensorId    = concatTensors[1].tensorId;
    uint64_t outputTensorId = concatTensors[2 * NUM_OF_LAUNCHES].tensorId;
    uint64_t shapeTensorId  = concatTensors[(2 * NUM_OF_LAUNCHES) + NUM_OF_LAUNCHES].tensorId;

    for (unsigned i = 0; i < NUM_OF_LAUNCHES; i++)
    {
        // Launch same recipe with different tensors
        std::vector<synLaunchTensorInfoExt> multiLaunchConcatTensors;
        multiLaunchConcatTensors.push_back(concatTensors[2 * i]);
        multiLaunchConcatTensors[0].tensorId = inputTensorId;
        multiLaunchConcatTensors.push_back(concatTensors[(2 * i) + 1]);
        multiLaunchConcatTensors[1].tensorId = h2dTensorId;
        multiLaunchConcatTensors.push_back(concatTensors[(2 * NUM_OF_LAUNCHES) + i]);
        multiLaunchConcatTensors[2].tensorId = outputTensorId;
        multiLaunchConcatTensors.push_back(concatTensors[(2 * NUM_OF_LAUNCHES) + NUM_OF_LAUNCHES + i]);
        multiLaunchConcatTensors[3].tensorId = shapeTensorId;

        status = synLaunchExt(m_streamHandleCompute,
                           multiLaunchConcatTensors.data(),
                           multiLaunchConcatTensors.size(),
                           programAddress,
                           graphData.recipeHandle,
                           0);
        ASSERT_EQ(status, synSuccess) << "Unexpected status for synLaunch";
    }

    status = synStreamSynchronize(m_streamHandleCompute);
    ASSERT_EQ(status, synSuccess) << "Failed to sync compute stream";

    copyOutputTensors(graphIndex);

    for (unsigned i = 0; i < NUM_OF_LAUNCHES; i++)
    {
        float* outputData = (float*)m_hostBuffers[out[i]];
        float* inputData  = (float*)m_hostBuffers[in[i]];

        unsigned outIndex = 0;
        for (unsigned l = 0; l < outActualSizes[i][0]; l++)
        {
            unsigned inIndex = l * stridesData[0].strides[0] + stridesData[0].offset;
            ASSERT_EQ(inputData[inIndex], outputData[outIndex])
                << "Mismatch at index " << l << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
            outIndex++;
        }
    }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_insert_dynamic_fcd_dynamic_param, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned originalSizesMax[NUM_DIMS]   = {8, 8};
    unsigned originalSizesMin[NUM_DIMS]   = {4, 4};
    unsigned insertSizesMax[NUM_DIMS - 1] = {(8 * 8) / 2};
    unsigned insertSizesMin[NUM_DIMS - 1] = {(4 * 4) / 2};

    unsigned stridesSize[NUM_DIMS - 1] = {2};
    unsigned offsetSize[1]             = {0};

    unsigned inReal = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          originalSizesMax,
                                          NUM_DIMS,
                                          syn_type_single,
                                          nullptr,
                                          "input_real",
                                          0,
                                          0,
                                          nullptr,
                                          originalSizesMin);

    unsigned inView = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          insertSizesMax,
                                          NUM_DIMS - 1,
                                          syn_type_single,
                                          nullptr,
                                          "input_view",
                                          0,
                                          0,
                                          nullptr,
                                          insertSizesMin);

    unsigned viewTensor = createTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       insertSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       insertSizesMin);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       originalSizesMax,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       originalSizesMin);

    unsigned stride = createShapeTensor(INPUT_TENSOR, stridesSize, stridesSize, NUM_DIMS - 1);
    unsigned offset = createShapeTensor(INPUT_TENSOR, offsetSize, offsetSize, 1);

    addNodeToGraph("relu_fwd_f32", {inView}, {viewTensor}, nullptr, 0, "relu");
    addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                   {inReal, viewTensor, stride, offset},
                   {out},
                   nullptr,
                   0,
                   "strided_insert");

    compileTopology();

    unsigned originalActualSizes[] = {6, 6};
    unsigned insertActualSizes[]   = {(6 * 6) / 2};
    setActualSizes(inReal, originalActualSizes);
    setActualSizes(inView, insertActualSizes);
    setActualSizes(out, originalActualSizes);

    runTopology();

    float* outputData    = (float*)m_hostBuffers[out];
    float* inputViewData = (float*)m_hostBuffers[inView];
    float* inputData     = (float*)m_hostBuffers[inReal];

    for (unsigned i = 0; i < insertActualSizes[0]; i++)
    {
        float expected = (i % 2 == 0) ? std::max(inputViewData[i / 2], 0.0f) : inputData[i];
        ASSERT_EQ(expected, outputData[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << outputData[i];
    }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_insert_dynamic_fcd_dynamic_param_with_h2d, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned originalSizesMax[NUM_DIMS]   = {8, 8};
    unsigned originalSizesMin[NUM_DIMS]   = {4, 4};
    unsigned insertSizesMax[NUM_DIMS - 1] = {(8 * 8) / 2};
    unsigned insertSizesMin[NUM_DIMS - 1] = {(4 * 4) / 2};

    synDynamicStridedDmaH2dTensor stridesData[2] = {{1, 0, {2, 1, 1, 1, 1}}, {1, 0, {2, 1, 1, 1, 1}}};
    unsigned                         stridesSize[1] = {sizeof(synDynamicStridedDmaH2dTensor) / sizeof(unsigned)};

    unsigned inReal = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          originalSizesMax,
                                          NUM_DIMS,
                                          syn_type_single,
                                          nullptr,
                                          "input_real",
                                          0,
                                          0,
                                          nullptr,
                                          originalSizesMin);

    unsigned inView = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          insertSizesMax,
                                          NUM_DIMS - 1,
                                          syn_type_single,
                                          nullptr,
                                          "input_view",
                                          0,
                                          0,
                                          nullptr,
                                          insertSizesMin);

    unsigned viewTensor = createTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       insertSizesMax,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       insertSizesMin);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       originalSizesMax,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       originalSizesMin);

    unsigned stride = createHost2DeviceTensor(INPUT_TENSOR, stridesSize, (unsigned *)stridesData, 1, "strides");

    addNodeToGraph("relu_fwd_f32", {inView}, {viewTensor}, nullptr, 0, "relu");
    addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                   {inReal, viewTensor, stride},
                   {out},
                   nullptr,
                   0,
                   "strided_insert");

    compileTopology();

    unsigned originalActualSizes[] = {6, 6};
    unsigned insertActualSizes[]   = {(6 * 6) / 2};
    setActualSizes(inReal, originalActualSizes);
    setActualSizes(inView, insertActualSizes);
    setActualSizes(out, originalActualSizes);
    setActualScalarParametersData(stride, stridesData, sizeof(synDynamicStridedDmaH2dTensor));

    runTopology();

    float* outputData    = (float*)m_hostBuffers[out];
    float* inputViewData = (float*)m_hostBuffers[inView];
    float* inputData     = (float*)m_hostBuffers[inReal];

    for (unsigned i = 0; i < insertActualSizes[0]; i++)
    {
        float expected = (i % 2 == 0) ? std::max(inputViewData[i / 2], 0.0f) : inputData[i];
        ASSERT_EQ(expected, outputData[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << outputData[i];
    }
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_view_zst, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned inSizes[NUM_DIMS]      = {8, 0};
    unsigned outSizes[NUM_DIMS - 1] = {0};

    unsigned stridesSize[NUM_DIMS - 1] = {2};
    unsigned offsetSize[1]             = {0};

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input");

    unsigned shape  = createShapeTensor(INPUT_TENSOR, outSizes, outSizes, NUM_DIMS - 1);
    unsigned stride = createShapeTensor(INPUT_TENSOR, stridesSize, stridesSize, NUM_DIMS - 1);
    unsigned offset = createShapeTensor(INPUT_TENSOR, offsetSize, offsetSize, 1);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizes,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       "out");
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName,
                   {in, shape, stride, offset},
                   {out},
                   nullptr,
                   0,
                   "strided_view");

    compileTopology();
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_view_zst_out, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned inSizes[NUM_DIMS]      = {8, 8};
    unsigned outSizes[NUM_DIMS - 1] = {0};

    synStridedOpParams params = {0};
    params.strides[0]         = 2;

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input");

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizes,
                                       NUM_DIMS - 1,
                                       syn_type_single,
                                       nullptr,
                                       "out");
    addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {in}, {out}, &params, sizeof(params), "strided_view");

    compileTopology();
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_insert_zst, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 2;

    unsigned inSizes[NUM_DIMS]       = {8, 8};
    unsigned viewSizes[NUM_DIMS - 1] = {0};

    synStridedOpParams params = {0};
    params.strides[0]         = 2;

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "input");

    unsigned view = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        viewSizes,
                                        NUM_DIMS - 1,
                                        syn_type_single,
                                        nullptr,
                                        "insert");

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       inSizes,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out");

    addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                   {in, view},
                   {out},
                   &params,
                   sizeof(params),
                   "strided_insert");

    compileTopology();
}

TEST_F_GC(SynGaudiStridedViewDynamicTest, strided_insert_producer_dep, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    static const unsigned NUM_DIMS = 1;

    unsigned sizes[] = {4};

    synStridedOpParams params;
    memset(&params, 0, sizeof(params));
    params.baseOffset = 0;
    params.strides[0] = 1;

    unsigned in1 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       sizes,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "input1");

    unsigned in2 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       sizes,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "input2");

    unsigned t1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, NUM_DIMS);
    unsigned t2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, NUM_DIMS);

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       sizes,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out");

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {in1}, {t1}, nullptr, 0, "memcopy");
    addNodeToGraph("add_f32", {in2, t1}, {t2}, nullptr, 0, "add");
    addNodeToGraph(NodeFactory::stridedInsertNodeTypeName, {t2, t1}, {out}, &params, sizeof(params), "strided_insert");

    compileTopology();
    runTopology();

    float* outputData = (float*)m_hostBuffers[out];
    float* inData1    = (float*)m_hostBuffers[in1];
    for (unsigned i = 0; i < sizes[0]; i++)
    {
        ASSERT_EQ(inData1[i], outputData[i])
            << "Mismatch at index " << i << " Expected: " << inData1[i] << " Result: " << outputData[i];
    }
}

/*
         +-------+     +-------+     +-------+     +-------+     +-------+     +-------+
  t1     |       | t3  |       | t4  |       | t5  |       | t6  |       | t7  |       |  t8
+-----+->+ View1 +---->+  Add1 +---->+Insert1+--+->+ View2 +---->+ Add2  +---->+Insert1+------->
      |  |       |     |       |     |       |  |  |       |     |       |     |       |
      |  +-------+     +-------+     +---+---+  |  +-------+     +-------+     +---+---+
      |                                  ^      |                                  ^
      |                                  |      |                                  |
      +----------------------------------+      +----------------------------------+

*/
TEST_F_GC(SynTrainingTestInfra, stridedInsertViewReusePersistentMem)
{
    static const unsigned NUM_DIMS             = 3;
    unsigned              inSizes[NUM_DIMS]    = {4, 3, 2};
    unsigned              sliceSize1[NUM_DIMS] = {4, 2, 2};
    unsigned              sliceSize2[NUM_DIMS] = {4, 1, 2};
    unsigned              numElements          = inSizes[0] * inSizes[1] * inSizes[2];

    unsigned sectionIndex = createSection(numElements * sizeof(float));

    unsigned tAdd1 = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sliceSize1,
                                         NUM_DIMS,
                                         syn_type_single,
                                         nullptr,
                                         "tAdd1");
    unsigned tAdd2 = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         sliceSize2,
                                         NUM_DIMS,
                                         syn_type_single,
                                         nullptr,
                                         "tAdd2");
    unsigned t1    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "in",
                                      0,
                                      0,
                                      &sectionIndex);
    unsigned t3    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize1, NUM_DIMS, syn_type_single);
    unsigned t4    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize1, NUM_DIMS, syn_type_single);
    unsigned t5    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inSizes, NUM_DIMS, syn_type_single);
    unsigned t6    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize2, NUM_DIMS, syn_type_single);
    unsigned t7    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sliceSize2, NUM_DIMS, syn_type_single);
    unsigned t8    = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      inSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "out",
                                      0,
                                      0,
                                      &sectionIndex);

    synStridedOpParams view1Params = {0};
    view1Params.strides[0]         = 1;
    view1Params.strides[1]         = view1Params.strides[0] * inSizes[0];
    view1Params.strides[2]         = view1Params.strides[1] * inSizes[1];

    synStridedOpParams view2Params = view1Params;
    view2Params.baseOffset         = 4;

    addNodeToGraph("strided_view", {t1}, {t3}, &view1Params, sizeof(view1Params), "view1");
    addNodeToGraph("add_fwd_f32", {t3, tAdd1}, {t4}, nullptr, 0, "add1");
    addNodeToGraph("strided_insert", {t1, t4}, {t5}, &view1Params, sizeof(view1Params), "insert1");
    addNodeToGraph("strided_view", {t5}, {t6}, &view2Params, sizeof(view1Params), "view2");
    addNodeToGraph("add_fwd_f32", {t6, tAdd2}, {t7}, nullptr, 0, "add2");
    addNodeToGraph("strided_insert", {t5, t7}, {t8}, &view2Params, sizeof(view2Params), "insert2");

    compileTopology();

    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 2);  // verify no internal memcopy nodes were added

    runTopology();

    float* inputData = (float*)m_hostBuffers[t1];
    float* add1Data  = (float*)m_hostBuffers[tAdd1];
    float* add2Data  = (float*)m_hostBuffers[tAdd2];

    // calculate expected output
    float expected[numElements];
    memcpy(expected, inputData, inSizes[0] * inSizes[1] * inSizes[2] * sizeof(float));

    // add1
    unsigned idx = 0;
    for (unsigned h = 0; h < sliceSize1[2]; h++)
    {
        for (unsigned w = 0; w < sliceSize1[1]; w++)
        {
            for (unsigned c = 0; c < sliceSize1[0]; c++)
            {
                unsigned offset = view1Params.baseOffset;
                offset += view1Params.strides[0] * c;
                offset += view1Params.strides[1] * w;
                offset += view1Params.strides[2] * h;
                expected[offset] += add1Data[idx++];
            }
        }
    }

    // add2
    idx = 0;
    for (unsigned h = 0; h < sliceSize2[2]; h++)
    {
        for (unsigned w = 0; w < sliceSize2[1]; w++)
        {
            for (unsigned c = 0; c < sliceSize2[0]; c++)
            {
                unsigned offset = view2Params.baseOffset;
                offset += view2Params.strides[0] * c;
                offset += view2Params.strides[1] * w;
                offset += view2Params.strides[2] * h;
                expected[offset] += add2Data[idx++];
            }
        }
    }

    float* outputData = (float*)m_hostBuffers[t8];
    // compare results
    for (unsigned i = 0; i < numElements; i++)
    {
        ASSERT_EQ(expected[i], outputData[i])
            << "Mismatch at index " << i << " Expected: " << expected[i] << " Result: " << outputData[i];
    }
}

/*
         +--------+                                               +--------+
   in1   |        |                                               |        |  out1
+------->+  add1  +----------------+                           +->+  SV1   +------->
         |        |                |                           |  |        |
         +---^----+           +----v---+          +--------+   |  +--------+
   in3       |         in0    |        |          |        |   |
+------------+    +---------->+  SI1   +---------->  SI2   +---+
             |                |        |          |        |   |
         +---v----+           +--------+          +---^----+   |  +--------+
   in2   |        |                                   |        |  |        |  out2
+------->+  add2  +-----------------------------------+        +->+  SV2   +------->
         |        |                                               |        |
         +--------+                                               +--------+
SI1 has the same strided params as SV1.
SI2 has the same strided params as SV2.

additional 3 independent configurations:
 - isStridedAccess - SI and SV will write/read in a sparse manner.
 - viewFromIntermediate - SV1 will read out of SI1 output
 - isOverlap - SI1 and SV1 will have different (overlapping) access
*/

class SynTrainingStridedViewMoveTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<bool /* strided access or not */,
                                                bool /* strided view from output or middle */,
                                                bool /* is overlap */>>
{
};

TEST_P_GC(SynTrainingStridedViewMoveTest, strided_view_move_test)
{
    ScopedConfigurationChange optimizeSi("ENABLE_OPTIMIZE_STRIDED_INSERT", "true");

    bool     isStridedAccess      = std::get<0>(GetParam());
    bool     viewFromIntermediate = std::get<1>(GetParam());
    bool     isOverlap            = std::get<2>(GetParam());
    unsigned viewSizes[]          = {2, 2};
    unsigned realSizes[]          = {8};

    unsigned section       = createSection(static_cast<uint64_t>(realSizes[0]) * 4);
    unsigned sectionOffset = multiplyElements(viewSizes, viewSizes + 2);

    unsigned in0  = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       realSizes,
                                       1,
                                       syn_type_single,
                                       nullptr,
                                       "in0",
                                       0,
                                       0,
                                       &section);
    unsigned in1  = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       viewSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       "in1");
    unsigned in2  = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       viewSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       "in2");
    unsigned in3  = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       viewSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       "in3");
    unsigned out1 = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        viewSizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        "out1",
                                        0,
                                        0,
                                        &section);
    unsigned out2 = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        viewSizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        "out2",
                                        0,
                                        sectionOffset * 4,
                                        &section);

    unsigned add1Out = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, viewSizes, 2);
    unsigned add2Out = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, viewSizes, 2);
    unsigned si1Out  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, realSizes, 1);
    unsigned si2Out  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, realSizes, 1);

    synNodeId          siId, sv1Id, sv2Id;
    synStridedOpParams si1Params = {0};
    si1Params.strides[0]         = isStridedAccess ? 2 : 1;
    si1Params.strides[1]         = viewSizes[0] * si1Params.strides[0];

    synStridedOpParams si2Params = si1Params;
    si2Params.baseOffset         = isStridedAccess ? 1 : sectionOffset;

    synStridedOpParams sv1Params = si1Params;
    synStridedOpParams sv2Params = si2Params;

    if (isOverlap)
    {
        si1Params.baseOffset = 1;
        si1Params.strides[0] = 1;
        si1Params.strides[1] = viewSizes[0] * si1Params.strides[0];
    }

    unsigned sv1Input = viewFromIntermediate ? si1Out : si2Out;
    addNodeToGraph("add_fwd_f32", {in1, in3}, {add1Out}, nullptr, 0, "add1");
    addNodeToGraph("add_fwd_f32", {in2, in3}, {add2Out}, nullptr, 0, "add2");
    addNodeToGraph("strided_insert", {in0, add1Out}, {si1Out}, &si1Params, sizeof(si1Params), "si1", 0, &siId);
    addNodeToGraph("strided_insert", {si1Out, add2Out}, {si2Out}, &si2Params, sizeof(si2Params), "si2");
    addNodeToGraph("strided_view", {sv1Input}, {out1}, &sv1Params, sizeof(sv1Params), "sv1", 0, &sv1Id);
    addNodeToGraph("strided_view", {si2Out}, {out2}, &sv2Params, sizeof(sv2Params), "sv2", 0, &sv2Id);

    setNodeDependency(&siId, &sv1Id, 1, 1);
    setNodeDependency(&siId, &sv2Id, 1, 1);

    compileAndRun();

    if (!isOverlap)  // check optimizeStridedInsert pass works as expected
    {
        // verify no internal memcopy nodes were added
        ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 2);
        // verify all SV and SI nodes are gone
        ASSERT_EQ(countNumViewInsertNodes(getGraph(0).graphHandle), 0);
    }
    else
    {
        // verify SV2/SI2 nodes are gone
        ASSERT_EQ(countNumViewInsertNodes(getGraph(0).graphHandle), 2);
    }

    const float* in0Data  = (const float*)m_hostBuffers[in0];
    const float* in1Data  = (const float*)m_hostBuffers[in1];
    const float* in2Data  = (const float*)m_hostBuffers[in2];
    const float* in3Data  = (const float*)m_hostBuffers[in3];
    const float* out1Data = (const float*)m_hostBuffers[out1];
    const float* out2Data = (const float*)m_hostBuffers[out2];

    std::vector<float> overlapOut1(4);
    if (isOverlap)
    {
        if (isStridedAccess)
        {
            overlapOut1 = {in0Data[0], in1Data[1] + in3Data[1], in1Data[3] + in3Data[3], in0Data[6]};
        }
        else if (!isStridedAccess)
        {
            overlapOut1 = {in0Data[0], in1Data[0] + in3Data[0], in1Data[1] + in3Data[1], in1Data[2] + in3Data[2]};
        }
    }

    for (int i = 0; i < sectionOffset; i++)
    {
        float expected = isOverlap ? overlapOut1[i] : in1Data[i] + in3Data[i];
        EXPECT_EQ(out1Data[i], expected) << "index " << i;
    }
    for (int i = 0; i < sectionOffset; i++)
    {
        float expected = in2Data[i] + in3Data[i];
        EXPECT_EQ(out2Data[i], expected) << "index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingStridedViewMoveTest,
                         ::testing::Combine(::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({true, false})));

// reproduce the issue from [SW-99711]
TEST_F_GC(SynTrainingTestInfra, stridedViewInPlaceTest)
{
    static const unsigned NUM_DIMS            = 4;
    unsigned              mulSizes[NUM_DIMS]  = {2, 2, 2, 2};
    unsigned              viewSizes[NUM_DIMS] = {32};
    unsigned              numElements         = viewSizes[0];

    unsigned sectionIndex = createSection(numElements * sizeof(float));

    unsigned t0 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      viewSizes,
                                      1,
                                      syn_type_single,
                                      nullptr,
                                      "t0",
                                      0,
                                      0,
                                      &sectionIndex);

    unsigned mul1In = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          mulSizes,
                                          NUM_DIMS,
                                          syn_type_single,
                                          nullptr,
                                          "mul1In");

    unsigned mul2In = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          mulSizes,
                                          NUM_DIMS,
                                          syn_type_single,
                                          nullptr,
                                          "mul2In");

    unsigned mul1ViewIn = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, mulSizes, NUM_DIMS, syn_type_single);
    unsigned mul1Out    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, mulSizes, NUM_DIMS, syn_type_single);
    unsigned mul2ViewIn = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, mulSizes, NUM_DIMS, syn_type_single);
    unsigned mul2Out    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, mulSizes, NUM_DIMS, syn_type_single);

    unsigned out1 = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        mulSizes,
                                        NUM_DIMS,
                                        syn_type_single,
                                        nullptr,
                                        "out1",
                                        0,
                                        0,
                                        &sectionIndex);

    unsigned out2 = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        mulSizes,
                                        NUM_DIMS,
                                        syn_type_single,
                                        nullptr,
                                        "out2",
                                        0,
                                        (numElements / 2) * sizeof(float),
                                        &sectionIndex);

    synStridedOpParams view1Params = {0};
    view1Params.strides[0]         = 1;
    view1Params.strides[1]         = view1Params.strides[0] * mulSizes[0];
    view1Params.strides[2]         = view1Params.strides[1] * mulSizes[1];
    view1Params.strides[3]         = view1Params.strides[2] * mulSizes[2];

    synStridedOpParams view2Params = view1Params;
    view2Params.baseOffset         = numElements / 2;

    synNodeId viewId[2];
    synNodeId outId[2];

    addNodeToGraph("strided_view", {t0}, {mul1ViewIn}, &view1Params, sizeof(view1Params), "view1", 0, &viewId[0]);
    addNodeToGraph("strided_view", {t0}, {mul2ViewIn}, &view2Params, sizeof(view2Params), "view2", 0, &viewId[1]);

    addNodeToGraph("mult_fwd_f32", {mul1ViewIn, mul1In}, {mul1Out}, nullptr, 0, "mul1");
    addNodeToGraph("mult_fwd_f32", {mul2ViewIn, mul2In}, {mul2Out}, nullptr, 0, "mul2");

    addNodeToGraph("identity", {mul1Out}, {out1}, nullptr, 0, "out1", 0, &outId[0]);
    addNodeToGraph("identity", {mul2Out}, {out2}, nullptr, 0, "out2", 0, &outId[1]);

    setNodeDependency(viewId, outId, 2, 2);

    compileAndRun();

    float* inputData  = (float*)m_hostBuffers[t0];
    float* mul1InData = (float*)m_hostBuffers[mul1In];
    float* mul2InData = (float*)m_hostBuffers[mul2In];
    float* out1Data   = (float*)m_hostBuffers[out1];
    float* out2Data   = (float*)m_hostBuffers[out2];

    // compare results
    for (unsigned i = 0; i < numElements / 2; i++)
    {
        float expected = inputData[i] * mul1InData[i];
        ASSERT_EQ(out1Data[i], expected) << "Mismatch at index " << i << " Expected: " << expected
                                         << " Result: " << out1Data[i];
    }
    for (unsigned i = 0; i < numElements / 2; i++)
    {
        float expected = inputData[i + numElements / 2] * mul2InData[i];
        ASSERT_EQ(out2Data[i], expected) << "Mismatch at index " << i << " Expected: " << expected
                                         << " Result: " << out1Data[i];
    }
}

// [SW-148087] remove redundant memcopy by replacing strided view/insert node with reshape
TEST_F_GC(SynTrainingTestInfra, replaceStridedViewWithReshape)
{
    static const unsigned NUM_DIMS              = 4;
    unsigned              realSizes[NUM_DIMS]   = {2, 2, 2, 2};
    unsigned              insertSizes[NUM_DIMS] = {4, 1, 2, 2};
    unsigned              viewSizes[NUM_DIMS]   = {4, 1, 4, 1};
    unsigned              numElements           = 16;

    unsigned t0 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      realSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "t0");

    unsigned t1 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      insertSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "t1");

    unsigned t2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, realSizes, NUM_DIMS, syn_type_single);

    unsigned t3 = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      viewSizes,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "t3");

    synStridedOpParams viewParams = {0};
    viewParams.strides[0]         = 1;
    viewParams.strides[1]         = viewParams.strides[0] * viewSizes[0];
    viewParams.strides[2]         = viewParams.strides[1] * viewSizes[1];
    viewParams.strides[3]         = viewParams.strides[2] * viewSizes[2];
    addNodeToGraph("strided_view", {t2}, {t3}, &viewParams, sizeof(viewParams), "view");

    synStridedOpParams insertParams = {0};
    insertParams.strides[0]         = 1;
    insertParams.strides[1]         = insertParams.strides[0] * insertSizes[0];
    insertParams.strides[2]         = insertParams.strides[1] * insertSizes[1];
    insertParams.strides[3]         = insertParams.strides[2] * insertSizes[2];
    addNodeToGraph("strided_insert", {t0, t1}, {t2}, &insertParams, sizeof(insertParams), "insert");

    compileAndRun();

    float* t1Data = castHostBuffer<float>(t1);
    float* t3Data = castHostBuffer<float>(t3);

    // compare results
    for (unsigned i = 0; i < numElements; i++)
    {
        ASSERT_EQ(t1Data[i], t3Data[i]) << "Mismatch at index " << i << " Expected: " << t1Data[i]
                                        << " Result: " << t3Data[i];
    }

    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 1);        // verify only 1 memcopy node was added
    ASSERT_EQ(countNumViewInsertNodes(getGraph(0).graphHandle), 0);  // verify all view nodes were removed
}
