#include <gtest/gtest.h>

#include "infra/memory_management/segment_alloc.hpp"

class UTSegment_mgr : public ::testing::Test
{
};

TEST_F(UTSegment_mgr, DISABLED_basic)
{
    const uint16_t N = 64;

    SegmentAlloc segmentAlloc;

    segmentAlloc.init(N);

    // After init, size should be N
    ASSERT_EQ(segmentAlloc.size(), N);
    ASSERT_EQ(segmentAlloc.isSanityChkOk(true), true);

    // allocate 5 segments
    auto segments = segmentAlloc.getSegments(5);
    ASSERT_EQ(segments.size(), 5);  // make sure got 5
    ASSERT_EQ(segmentAlloc.size(), N - 5);
    ASSERT_EQ(segmentAlloc.isSanityChkOk(false), true);

    // release back the 5 segments
    segmentAlloc.releaseSegments(segments);
    ASSERT_EQ(segmentAlloc.size(), N);
    ASSERT_EQ(segmentAlloc.isSanityChkOk(true), true);

    // release the same again,
    segmentAlloc.releaseSegments(segments);
    ASSERT_EQ(segmentAlloc.isSanityChkOk(false), false);  // should detect an error

    // alloc again to get to a good state
    segments = segmentAlloc.getSegments(5);
    ASSERT_EQ(segmentAlloc.isSanityChkOk(false), true);  // should detect an error
}

TEST_F(UTSegment_mgr, simulateRealWork)
{
    const uint16_t N = 64;

    SegmentAlloc segmentAlloc;
    segmentAlloc.init(N);

    std::deque<std::vector<uint16_t>> allocated;

    // allocate until we are full
    while (true)
    {
        auto x = segmentAlloc.getSegments(rand() % 5 + 1);
        if (x.size() == 0)
        {
            ASSERT_EQ(segmentAlloc.isSanityChkOk(false), true);
            break;
        }
        allocated.push_back(x);
    }

    // free and allocate again few times
    for (int i = 0; i < 10000; i++)
    {
        uint16_t needed = rand() % 6 + 1;

        // release until we have room
        while (segmentAlloc.size() < needed)
        {
            auto toRelease = allocated.front();
            allocated.pop_front();
            segmentAlloc.releaseSegments(toRelease);
        }

        // now allocate
        auto x = segmentAlloc.getSegments(needed);
        allocated.push_back(x);
    }

    // release all
    while (!allocated.empty())
    {
        auto x = allocated.back();
        allocated.pop_back();
        segmentAlloc.releaseSegments(x);
    }

    // now make sure everything is OK
    ASSERT_EQ(segmentAlloc.size(), N);
    ASSERT_EQ(segmentAlloc.isSanityChkOk(true), true);
}
