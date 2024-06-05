#include "graph_optimizer_test.h"
#include "include/sync/data_range.h"
#include "include/sync/overlap.h"
#include "gtest/gtest.h"

class OverlapTest : public GraphOptimizerTest
{
protected:
    static const int N = 6;

    OverlapDescriptor createTransaction(unsigned               engine,
                                        Overlap<N>::AccessType access,
                                        uint64_t               offset,
                                        uint32_t               stride,
                                        uint64_t               size,
                                        uint32_t               reps,
                                        bool                   useCyclic)
    {
        OverlapDescriptor desc;
        desc.engineID   = engine;
        desc.engineIDForDepCtx = desc.engineID;
        desc.numSignals = 1;
        auto& roiList   = (access == Overlap<N>::AccessType::READ) ? desc.inputRois : desc.outputRois;
        roiList.emplace_back();
        OverlapRoi& roi = roiList.back();
        roi.isReduction = (access == Overlap<N>::AccessType::RMW);
        roi.offset      = offset;
        roi.subRois->emplace_back();
        OverlapSubRoi& subRoi = roi.subRois->back();
        subRoi.relSoIdx       = 0;
        if (useCyclic)
        {
            subRoi.ranges.emplace_back(0, stride * reps);
            subRoi.cyclicRanges.emplace_back(0, size, stride);
        }
        else
        {
            for (int i = 0; i < reps; i++)
            {
                subRoi.ranges.emplace_back(i * stride, i * stride + size);
            }
        }
        return desc;
    }

    void compareCtx(const Overlap<N>::DependencyCtx& ctx1, const Overlap<N>::DependencyCtx& ctx2, uint64_t addr)
    {
        for (int i = 0; i < N; i++)
        {
            EXPECT_EQ(ctx1.valid[i], ctx2.valid[i]) << "addr: " << addr;
            if (ctx1.valid[i])
            {
                EXPECT_EQ(ctx1.signalIdx[i], ctx2.signalIdx[i]) << "addr: " << addr;
            }
        }
    }

    void compareSegmentSpaces(Overlap<N>& overlap1, Overlap<N>& overlap2, int numTests, uint64_t maxAddr)
    {
        /*
            statistic comparisson of segment space. using numTests X 1 byte random writes in [0, maxAddr]
        */
        for (int k = 0; k < numTests; k++)
        {
            Overlap<N>::DependencyCtx ctx1;
            Overlap<N>::DependencyCtx ctx2;
            uint64_t                  addr = rand() % maxAddr;
            OverlapDescriptor         desc = createTransaction(4, Overlap<N>::AccessType::WRITE, addr, 1, 1, 1, false);
            overlap1.addDescriptor(desc, ctx1);
            overlap2.addDescriptor(desc, ctx2);

            compareCtx(ctx1, ctx2, addr);
        }
    }
};

class CyclicOverlapTest : public OverlapTest
{
protected:
    Overlap<N> createOverlap(bool useCyclic)
    {
        /*
            Basic overlap test with writes, reads, and RMW.
            Compare the segment space created by using cyclic ranges,
            with the segment space created by using only linear ranges
        */
        static const int  numTransactions = 10;
        Overlap<N>        overlap;
        OverlapDescriptor desc[numTransactions];

        // Engine A prepares 512X1000 features tensor in 4 activations
        desc[0] = createTransaction(0, Overlap<N>::AccessType::WRITE, 0, 1000, 256, 512, useCyclic);
        desc[1] = createTransaction(0, Overlap<N>::AccessType::WRITE, 256, 1000, 256, 512, useCyclic);
        desc[2] = createTransaction(0, Overlap<N>::AccessType::WRITE, 512, 1000, 256, 512, useCyclic);
        desc[3] = createTransaction(0, Overlap<N>::AccessType::WRITE, 768, 1000, 232, 512, useCyclic);

        // Engine B reads a 400x200 region of the feature map after logical reshape
        desc[4] = createTransaction(1, Overlap<N>::AccessType::READ, 1100, 2000, 256, 200, useCyclic);
        desc[5] = createTransaction(1, Overlap<N>::AccessType::READ, 1356, 2000, 144, 200, useCyclic);

        // Engine B reads a 400x200 region of the feature map after logical reshape
        desc[6] = createTransaction(2, Overlap<N>::AccessType::READ, 201128, 1000, 256, 200, useCyclic);
        desc[7] = createTransaction(2, Overlap<N>::AccessType::READ, 201384, 1000, 144, 200, useCyclic);

        // Engine C updates the features tensor
        desc[8] = createTransaction(3, Overlap<N>::AccessType::RMW, 131072, 256, 128, 160, useCyclic);
        desc[9] = createTransaction(3, Overlap<N>::AccessType::RMW, 200128, 256, 120, 160, useCyclic);

        for (const auto& transaction : desc)
        {
            Overlap<N>::DependencyCtx ctx;
            overlap.addDescriptor(transaction, ctx);
        }

        return overlap;
    }
};

TEST_F(CyclicOverlapTest, test)
{
    Overlap<N> overlap       = createOverlap(false);
    Overlap<N> cyclicOverlap = createOverlap(true);

    static const int      numTests = 10000;
    static const uint64_t maxAddr  = 400000;

    compareSegmentSpaces(overlap, cyclicOverlap, numTests, maxAddr);
}

class CyclicOverlapTestRandom
: public CyclicOverlapTest
, public testing::WithParamInterface<int32_t>
{
};

TEST_P(CyclicOverlapTestRandom, test)
{
    /*
        overlap test using randomized transactions.
        After each transactions, compare the dependencies created by using cyclic ranges,
        with the dependencies created by using only linear ranges.
        After all transactions, compare the complete segment space in both cases (statistic comparisson)
    */
    static const int      NUM_TRANSACTIONS = 1500;
    static const int      NUM_ACCESS_TYPES = 3;
    static const uint64_t MAX_OFFSET       = 1024 * 1024 * 1024;
    static const int      NUM_STRIDE_TYPES = 6;  // number of strides possible + 1 linear + 1 random
    static const int      RANDOM_STRIDE    = 4;
    static const int      LINEAR           = 5;
    static const int      STRIDES[4]       = {128, 258, 512, 1024};
    static const uint64_t MAX_SIZE         = 4 * 1024;
    static const int      MAX_REPS         = 500;

    srand(GetParam());
    Overlap<N> overlap;
    Overlap<N> cyclicOverlap;

    for (int i = 0; i < NUM_TRANSACTIONS; i++)
    {
        unsigned engine = rand() % N;
        auto     access = (Overlap<N>::AccessType)(rand() % NUM_ACCESS_TYPES);
        uint64_t offset = rand() % MAX_OFFSET;
        uint32_t reps   = rand() % MAX_REPS;
        uint64_t size   = rand() % MAX_SIZE;

        int     strideSwitch = rand() % NUM_STRIDE_TYPES;
        int32_t stride;
        if (strideSwitch == RANDOM_STRIDE)
        {
            stride = rand() % 3000 + 1;
            size   = rand() % stride;
        }
        else if (strideSwitch == LINEAR)
        {
            stride = size;
            reps   = 1;
        }
        else  // pre-determined stride
        {
            stride = STRIDES[strideSwitch];
            size   = rand() % stride;
        }
        size = (size == 0) ? stride : size;

        OverlapDescriptor desc = createTransaction(engine, access, offset, stride, size, reps, false);
        OverlapDescriptor descCyclic =
            createTransaction(engine, access, offset, stride, size, reps, strideSwitch != LINEAR);

        Overlap<N>::DependencyCtx ctx;
        Overlap<N>::DependencyCtx cyclicCtx;

        overlap.addDescriptor(desc, ctx);
        cyclicOverlap.addDescriptor(descCyclic, cyclicCtx);

        compareCtx(ctx, cyclicCtx, i);
    }
    compareSegmentSpaces(overlap, cyclicOverlap, 100000, MAX_OFFSET + MAX_SIZE * MAX_REPS);
}

INSTANTIATE_TEST_SUITE_P(CyclicOverlapTest, CyclicOverlapTestRandom, ::testing::ValuesIn({42, 1, 8, 13}));
