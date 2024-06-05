#include "log_manager.h"
#include "settable.h"
#include "sram_management_fe_test.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "types.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <array>
#include <string>

namespace gaudi
{
struct BatchSizes
{
    unsigned min;
    unsigned max;
};
using BatchArr = std::array<BatchSizes, 2>;
struct BgemmTestParams
{
    bool                sharedInput;
    unsigned            M;
    unsigned            K;
    unsigned            N;
    BatchArr            in0Batch;
    std::array<bool, 2> in1BatchBroadcast;
};
class SRAMManagementBatchGemmTest
: public SRAMManagementTest
, public testing::WithParamInterface<std::tuple<bool,        // shared input
                                                unsigned,    // M
                                                unsigned,    // K
                                                unsigned,    // N
                                                BatchSizes,  // input0 batch #1
                                                BatchSizes,  // input0 batch #2
                                                bool,        // input1 batch1 broadcast
                                                bool         // input1 batch1 broadcast
                                                >>
{
protected:
    SRAMManagementBatchGemmTest()
    {
        m_testParams.sharedInput          = std::get<0>(GetParam());
        m_testParams.M                    = std::get<1>(GetParam());
        m_testParams.K                    = std::get<2>(GetParam());
        m_testParams.N                    = std::get<3>(GetParam());
        m_testParams.in0Batch[0]          = std::get<4>(GetParam());
        m_testParams.in0Batch[1]          = std::get<5>(GetParam());
        m_testParams.in1BatchBroadcast[0] = std::get<6>(GetParam());
        m_testParams.in1BatchBroadcast[1] = std::get<7>(GetParam());
    }

protected:
    void setSramSize(unsigned size) { setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, std::to_string(size)); }

    NodePtr    createBgemm(bool dynamic = false);
    void       addSharedInput(TensorPtr sharedInput);
    void validateBatchSize(bool sharedInput);
    void compareStaticVsDynamicSlicing();
    BatchSizes initIn1BatchSize(unsigned batchIndex);

    BgemmTestParams m_testParams;
};

class SRAMManagementDynamicBatchGemmTest : public SRAMManagementBatchGemmTest
{
};

NodePtr SRAMManagementBatchGemmTest::createBgemm(bool dynamic)
{
    BatchSizes         in1Batch0 {}, in1Batch1 {};
    std::vector<TSize> in0MinSizes {}, in1MinSizes {}, outMinSizes {};
    synGEMMParams params  = synGEMMParams();

    in1Batch0 = initIn1BatchSize(0);
    in1Batch1 = initIn1BatchSize(1);

    std::vector<TSize> in0MaxSizes(
        {{m_testParams.K, m_testParams.M, m_testParams.in0Batch[0].max, m_testParams.in0Batch[1].max}});
    std::vector<TSize> in1MaxSizes({m_testParams.N, m_testParams.K, in1Batch0.max, in1Batch1.max});
    std::vector<TSize> outMaxSizes(
        {m_testParams.N, m_testParams.M, m_testParams.in0Batch[0].max, m_testParams.in0Batch[1].max});

    if (dynamic)
    {
        in0MinSizes = std::vector<TSize>(
            {{m_testParams.K, m_testParams.M, m_testParams.in0Batch[0].min, m_testParams.in0Batch[1].min}});
        in1MinSizes = std::vector<TSize>({m_testParams.N, m_testParams.K, in1Batch0.min, in1Batch1.min});
        outMinSizes = std::vector<TSize>(
            {m_testParams.N, m_testParams.M, m_testParams.in0Batch[0].min, m_testParams.in0Batch[1].min});
    }
    else
    {
        in0MinSizes = in0MaxSizes;
        in1MinSizes = in1MaxSizes;
        outMinSizes = outMaxSizes;
    }

    TensorPtr in1  = createTensor(in0MaxSizes, syn_type_float, true, in0MinSizes);
    TensorPtr in2  = createTensor(in1MaxSizes, syn_type_float, true, in1MinSizes);
    TensorPtr out1 = createTensor(outMaxSizes, syn_type_float, true, outMinSizes);

    auto bundle = createSingleMMENodeBundle({in1, in2}, {out1}, "batch_gemm", (void*)&params, sizeof(params));

    return bundle->getNodes().front();
}

void SRAMManagementBatchGemmTest::addSharedInput(TensorPtr sharedInput)
{
    synGEMMParams      params = synGEMMParams();
    std::vector<TSize> inMaxSizes(
        {{m_testParams.K, m_testParams.M, m_testParams.in0Batch[0].max, m_testParams.in0Batch[1].max}});
    std::vector<TSize> outMaxSizes(
        {m_testParams.N, m_testParams.M, m_testParams.in0Batch[0].max, m_testParams.in0Batch[1].max});

    TensorPtr in  = createTensor(inMaxSizes, syn_type_float);
    TensorPtr out = createTensor(outMaxSizes, syn_type_float);

    createSingleMMENodeBundle({in, sharedInput}, {out}, "batch_gemm", (void*)&params, sizeof(params));
}

BatchSizes SRAMManagementBatchGemmTest::initIn1BatchSize(unsigned batchIndex)
{
    BatchSizes sizes {};

    if (m_testParams.in1BatchBroadcast[batchIndex])
    {
        sizes.min = sizes.max = 1;
    }
    else
    {
        sizes.min = m_testParams.in0Batch[batchIndex].min;
        sizes.max = m_testParams.in0Batch[batchIndex].max;
    }
    return sizes;
}

void SRAMManagementBatchGemmTest::validateBatchSize(bool sharedInput)
{
    uint64_t           batchSizeCounter = 0;
    uint64_t           slicesCounter    = 0;
    Settable<unsigned> slicingDim;
    unsigned           maxChunkSize = 0;
    unsigned           minChunkSize = m_testParams.in0Batch[0].max * m_testParams.in0Batch[1].max;

    for (const NodePtr& node : getGraph().getNodes())
    {
        if (node->isBatchGemm())
        {
            ++slicesCounter;
            TensorPtr output = node->getOutput(0);
            unsigned  actual = 1;
            for (unsigned dim = DIM_GEMM_BATCH; dim < output->getDim(); ++dim)
            {
                actual *= output->getSizeInElements(dim);
                unsigned chunkSize = output->getSizeInElements(dim);
                // check that there is only 1 dim that: 1 < chunk size < final shape
                if (slicingDim.is_set())
                {
                    ASSERT_TRUE(slicingDim.value() == dim ||                                    // is slicing on dim
                                chunkSize == 1 ||                                               //
                                chunkSize == m_testParams.in0Batch[dim - DIM_GEMM_BATCH].max);  // full size
                }
                else if (chunkSize != m_testParams.in0Batch[dim - DIM_GEMM_BATCH].max)
                {
                    slicingDim.set(dim);
                }
                // on the slicing dim find the max and min chunk size
                if (slicingDim.is_set() && slicingDim.value() == dim)
                {
                    maxChunkSize = std::max(chunkSize, maxChunkSize);
                    minChunkSize = std::min(chunkSize, minChunkSize);
                }
            }
            batchSizeCounter += actual;
        }
    }
    ASSERT_EQ(batchSizeCounter,
              (sharedInput) ? 2 * m_testParams.in0Batch[0].max * m_testParams.in0Batch[1].max
                            : m_testParams.in0Batch[0].max * m_testParams.in0Batch[1].max);
    if (slicingDim.is_set())
    {
        unsigned numOfSlicesOnSlicingDim =
            slicesCounter / ((slicingDim.value() == DIM_GEMM_BATCH) ? m_testParams.in0Batch[1].max : 1);
        numOfSlicesOnSlicingDim /= (sharedInput) ? 2 : 1;
        // slice size is not too small
        ASSERT_TRUE(maxChunkSize - minChunkSize < numOfSlicesOnSlicingDim);
        // slice size is not too high
        ASSERT_TRUE(maxChunkSize <=
                    1 + m_testParams.in0Batch[slicingDim.value() - DIM_GEMM_BATCH].max / numOfSlicesOnSlicingDim);
    }
}

TEST_P(SRAMManagementBatchGemmTest, mme_batch_gemm_test)
{
    bool sharedInput = m_testParams.sharedInput;
    setSramSize((1 << 20) * 4);

    auto bgemm = createBgemm();

    if (sharedInput)
    {
        auto sharedInput = bgemm->getInput(1);
        addSharedInput(sharedInput);
    }

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    validateBatchSize(sharedInput);
}

INSTANTIATE_TEST_SUITE_P(batch_gemm_test,
                         SRAMManagementBatchGemmTest,
                         ::testing::Combine(::testing::Values(true, false),       // shared input
                                            ::testing::Values(4, 17, 93),         // M
                                            ::testing::Values(13, 55, 87),        // K
                                            ::testing::Values(19, 73, 101),       // N
                                            ::testing::Values(BatchSizes {8, 8},  // input0 batch #1 (min,max)
                                                              BatchSizes {15, 15},
                                                              BatchSizes {23, 23},
                                                              BatchSizes {33, 33}),
                                            ::testing::Values(BatchSizes {7, 7},  // input0 batch #1 (min,max)
                                                              BatchSizes {16, 16},
                                                              BatchSizes {27, 27},
                                                              BatchSizes {42, 42}),
                                            ::testing::Values(false),  // input1 batch #1 broadcast
                                            ::testing::Values(false)   // input1 batch #2 broadcast
                                            ));

TEST_P(SRAMManagementDynamicBatchGemmTest, compare_static_to_dynamic)
{
    createBgemm(false);
    const auto nodes = getGraph().getNodes();

    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    auto numOfMMENodesInStatic =
        std::count_if(nodes.cbegin(), nodes.cend(), [](NodePtr node) { return HabanaGraph::runsOnMME(node); });

    getGraph().clear();

    createBgemm(true);
    ASSERT_TRUE(sliceGraphToSRAMCapacity(getGraph()));
    auto numOfMMENodesInDynamic =
        std::count_if(nodes.cbegin(), nodes.cend(), [](NodePtr node) { return HabanaGraph::runsOnMME(node); });

    ASSERT_EQ(numOfMMENodesInStatic, numOfMMENodesInDynamic);
}

// Verfies that slicer slices the same number of nodes for static and dynamic
INSTANTIATE_TEST_SUITE_P(dynamic_bgemm_slicing_test,
                         SRAMManagementDynamicBatchGemmTest,
                         ::testing::Values(std::make_tuple(false,                // shared input
                                                           4,                    // M
                                                           13,                   // K
                                                           19,                   // N
                                                           BatchSizes {32, 32},  // input0 batch #1 (min,max)
                                                           BatchSizes {32, 16},  // input0 batch #2 (min,max)
                                                           true,                 // input1 batch #1 broadcast
                                                           true                  // input1 batch #2 broadcast
                                                           ))                    // min input1 batch
);

}  // namespace gaudi