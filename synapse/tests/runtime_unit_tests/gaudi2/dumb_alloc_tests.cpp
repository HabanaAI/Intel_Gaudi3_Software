#include "runtime/scal/common/recipe_launcher/dumb_allocator.hpp"

#include <gtest/gtest.h>

class UTGaudi2DumbAllocTest : public ::testing::Test
{
};

TEST_F(UTGaudi2DumbAllocTest, dumbAllocBasicTest)
{
    uint8_t* startAddr = (uint8_t*)0x20000;
    uint64_t size      = 0x10000;
    uint32_t numChunks = 8;

    DumbAllocator alloc(startAddr, size, numChunks);
    uint8_t*      addrArr[numChunks];

    for (int i = 0; i < numChunks; i++)
    {
        addrArr[i] = alloc.allocate(size / numChunks);

        ASSERT_EQ(addrArr[i], startAddr + (i * size / numChunks)) << "Wrong addr";
    }

    uint8_t* addr = alloc.allocate(size / numChunks / 2);
    ASSERT_EQ(addr, nullptr);

    for (int i = 3; i < 6; i++)
    {
        alloc.free(addrArr[i]);
    }

    for (int i = 3; i < 6; i++)
    {
        addrArr[i] = alloc.allocate(size / numChunks);
        ASSERT_EQ(addrArr[i], startAddr + (i * size / numChunks)) << "Wrong addr";
    }

    for (int i = 0; i < numChunks; i++)
    {
        alloc.free(addrArr[i]);
    }
}
