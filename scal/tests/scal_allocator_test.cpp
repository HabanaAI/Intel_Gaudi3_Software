#include <gtest/gtest.h>
#include <math.h>
#include "scal.h"
#include "scal_basic_test.h"
#include "common/scal_allocator.h"
#include "logger.h"

const uint64_t Scal::Allocator::c_bad_alloc; // needs to be defined at tests library

class ScalHeapAllocatorTests : public SCALTest {};

TEST_F_CHKDEV(ScalHeapAllocatorTests, no_reuse,{ALL})
{
    unsigned size = 0x10;
    ScalHeapAllocator alloc("SLAB_TEST");
    alloc.setSize(0x1000);

    uint64_t p1 = alloc.alloc(size, 128);
    uint64_t p2 = alloc.alloc(size, 128);
    ASSERT_NE(p1, Scal::Allocator::c_bad_alloc);
    ASSERT_NE(p2, Scal::Allocator::c_bad_alloc);
    ASSERT_NE(p1, p2) << "Invalid memory reuse";

    alloc.free(p1);
    uint64_t p3 = alloc.alloc(size, 128);
    ASSERT_NE(p3, Scal::Allocator::c_bad_alloc);

    ASSERT_EQ(p1, p3) << "No memory reuse";

    alloc.free(p2);
    uint64_t p4 = alloc.alloc(size, 160);
    ASSERT_EQ(p4, 160U);

    uint64_t p5 = alloc.alloc(size, 50);
    ASSERT_EQ(p5, 50U);

    uint64_t p6 = alloc.alloc(300, 50);
    ASSERT_EQ(p6, 200U);

    uint64_t p7 = alloc.alloc(5, 5);
    ASSERT_EQ(p7, 20U);

    uint64_t p8 = alloc.alloc(20, 7);
    ASSERT_EQ(p8, 28U);

    uint64_t p9 = alloc.alloc(5, 5);
    ASSERT_EQ(p9, 70U);

    alloc.free(p3);
    alloc.free(p4);
    alloc.free(p5);
    alloc.free(p6);
    alloc.free(p7);
    alloc.free(p8);
    alloc.free(p9);

    uint64_t p10 = alloc.alloc(0x1000, 128);
    ASSERT_EQ(p10, 0U);

    uint64_t p11 = alloc.alloc(1);
    ASSERT_EQ(p11, Scal::Allocator::c_bad_alloc);

    alloc.free(p10);
    uint64_t p12 = alloc.alloc(0x1001);
    ASSERT_EQ(p12, Scal::Allocator::c_bad_alloc);

    uint64_t p13 = alloc.alloc(5);
    uint64_t p14 = alloc.alloc(5);
    uint64_t p15 = alloc.alloc(5);
    ASSERT_EQ(p13, 0U);
    ASSERT_EQ(p14, 5U);
    ASSERT_EQ(p15, 10U);
}

TEST_F_CHKDEV(ScalHeapAllocatorTests, smaller_than_alignment_64,{ALL})
{
    unsigned size               = 0x10;
    uint64_t alignment          = 64;
    unsigned alloc_size         = 0x1000;
    ScalHeapAllocator alloc("SLAB_TEST");
    alloc.setSize(alloc_size);

    uint64_t old_alloc = -alignment;

    while (alloc_size > 0)
    {
        uint64_t new_alloc = alloc.alloc(size, alignment);
        ASSERT_NE(new_alloc, Scal::Allocator::c_bad_alloc);
        ASSERT_EQ(new_alloc, old_alloc + alignment) << "Invalid memory allocation";
        old_alloc = new_alloc;
        alloc_size -= alignment;
    }
}

TEST_F_CHKDEV(ScalHeapAllocatorTests, bigger_than_alignment_64,{ALL})
{
    uint64_t alignment  = 64;
    unsigned size               = alignment + 0x10;
    unsigned alloc_size         = 0x1000;
    ScalHeapAllocator alloc("SLAB_TEST");
    alloc.setSize(alloc_size);

    uint64_t old_alloc = -(alignment * 2);

    while (alloc_size > 0)
    {
        uint64_t new_alloc = alloc.alloc(size, alignment);
        ASSERT_NE(new_alloc, Scal::Allocator::c_bad_alloc);
        ASSERT_EQ(new_alloc, old_alloc + (alignment * 2)) << "Invalid memory allocation";
        old_alloc = new_alloc;
        alloc_size -= (alignment * 2);
    }
}
