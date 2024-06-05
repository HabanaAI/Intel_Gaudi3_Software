#include "runtime/scal/common/recipe_launcher/nbuff_allocator.hpp"

#include <gtest/gtest.h>
#include <synapse_api.h>
#include <scoped_configuration_change.h>
#include "global_statistics.hpp"
#include "test_utils.h"

class UTGaudi2nBuffAlloc : public ::testing::Test
{
public:
    static constexpr uint64_t CHUNK_SIZE = 100;
    void nBuffAllocTester(std::function<uint16_t()> func);
};

class AllocW : protected NBuffAllocator
{
public:
    AllocW() { NBuffAllocator::init(UTGaudi2nBuffAlloc::CHUNK_SIZE); }

    using NBuffAllocator::alloc;
    using NBuffAllocator::setLongSo;
    using NBuffAllocator::AllocRtn;
    using NBuffAllocator::dumpLongSo;
    using NBuffAllocator::init;
    using NBuffAllocator::getLastLongSo;
};

// allocate one buffer. Do it for NUM_BUFF * 2
TEST_F(UTGaudi2nBuffAlloc, nBuffAllocBasicTest1)
{
    AllocW allocator;
//    allocator.init(CHUNK_SIZE);

    constexpr uint16_t NUM_BUFF = NBuffAllocator::NUM_BUFF;

    uint64_t longSo = 1;

    // alloc one buff
    for (int i = 0; i < NUM_BUFF; i++)
    {
        AllocW::AllocRtn rtn = allocator.alloc(1 * CHUNK_SIZE);
        ASSERT_EQ(rtn.offset, i * CHUNK_SIZE);
        ASSERT_EQ(rtn.longSo, 0);
        allocator.setLongSo(longSo++);
    }
    // alloc one buff again, expect the right longSo
    for (int i = 0; i < NUM_BUFF; i++)
    {
        AllocW::AllocRtn rtn = allocator.alloc(1 * CHUNK_SIZE);
        ASSERT_EQ(rtn.offset, i * CHUNK_SIZE);
        ASSERT_EQ(rtn.longSo, i + 1);
        allocator.setLongSo(longSo++);
    }
}

// allocate one buffer, than two buffers every time for NUM_BUFF
TEST_F(UTGaudi2nBuffAlloc, nBuffAllocBasicTest2)
{
    AllocW allocator;

    constexpr uint16_t NUM_BUFF = NBuffAllocator::NUM_BUFF;

    uint64_t longSo = 1;
    // alloc one buffer
    {
        AllocW::AllocRtn rtn = allocator.alloc(1 * CHUNK_SIZE);
        ASSERT_EQ(rtn.offset, 0);
        ASSERT_EQ(rtn.longSo, 0);
        allocator.setLongSo(longSo++);
    }

    // alloc two buff every time until the half
    for (int i = 0; i < (NUM_BUFF - 1) / 2 / 2; i++)
    {
        AllocW::AllocRtn rtn = allocator.alloc(2 * CHUNK_SIZE);
        ASSERT_EQ(rtn.offset, (1 + i * 2) * CHUNK_SIZE);
        ASSERT_EQ(rtn.longSo, 0);
        allocator.setLongSo(longSo++);
    }

    // this should be allocated on the second half
    for (int i = 0; i < NUM_BUFF / 2 / 2; i++)  // /2->two buff per alloc, /2->fill one half
    {
        AllocW::AllocRtn rtn = allocator.alloc(2 * CHUNK_SIZE);
        ASSERT_EQ(rtn.offset, (NUM_BUFF / 2 + i * 2) * CHUNK_SIZE);
        ASSERT_EQ(rtn.longSo, 0);
        allocator.setLongSo(longSo++);
    }

    // alloc again all buff, 2 buffers each time
    for (int i = 0; i < NUM_BUFF / 2; i++)  // /2->two buff per alloc
    {
        AllocW::AllocRtn rtn = allocator.alloc(2 * CHUNK_SIZE);
        ASSERT_EQ(rtn.offset, (i * 2)  * CHUNK_SIZE);
        uint64_t expected = (i < (NUM_BUFF / 2 / 2 - 1)) ? i + 2 : i + 1;
        ASSERT_EQ(rtn.longSo, expected) << " i is " << i;
        allocator.setLongSo(longSo++);
    }
}

// allocate a lot of buffers (> N/2), should always allocate from start
TEST_F(UTGaudi2nBuffAlloc, nBuffAllocBig)
{
    ScopedEnvChange enableStat("ENABLE_STATS", "true");
    ScopedEnvChange expFlags("EXP_FLAGS", "true");
    ASSERT_EQ(synInitialize(), synSuccess);
    AllocW allocator;

    constexpr uint16_t NUM_BUFF = NBuffAllocator::NUM_BUFF;

    uint64_t      longSo = 1;
    constexpr int LOOPS  = 10000;
    for (int i = 0; i < LOOPS; i++)
    {
        uint16_t         numBuff = (NUM_BUFF / 2) + 1 + (rand() % (NUM_BUFF / 2));
        AllocW::AllocRtn rtn     = allocator.alloc(numBuff * CHUNK_SIZE);

        ASSERT_EQ(rtn.offset, 0);
        ASSERT_EQ(rtn.longSo, i);

        allocator.setLongSo(longSo++);
    }
    ASSERT_EQ(g_globalStat.get(globalStatPointsEnum::scalHbmSingleBuff).sum, LOOPS - 1);
    ASSERT_EQ(g_globalStat.get(globalStatPointsEnum::scalHbmSingleBuff).count, LOOPS - 1);

    ASSERT_EQ(synDestroy(), synSuccess);
}

TEST_F(UTGaudi2nBuffAlloc, nBuffDoubleAlloc)
{
    AllocW allocator;

    AllocW::AllocRtn rtn = allocator.alloc(1 * CHUNK_SIZE);
    ASSERT_EQ(rtn.offset, 0);
    ASSERT_EQ(rtn.longSo, 0);

    try
    {
        rtn = allocator.alloc(1 * CHUNK_SIZE); // second allocation without setLongSo, should throw
        UNUSED(rtn);
        FAIL() << "Expected throw";
    }
    catch(std::runtime_error const & err)
    {
    }
    catch(...)
    {
        FAIL() << "Expected throw";
    }
}

TEST_F(UTGaudi2nBuffAlloc, nBuffDoubleSetLongSo)
{
    AllocW allocator;

    AllocW::AllocRtn rtn = allocator.alloc(1 * CHUNK_SIZE);
    ASSERT_EQ(rtn.offset, 0);
    ASSERT_EQ(rtn.longSo, 0);

    allocator.setLongSo(1);

    try
    {
        allocator.setLongSo(1);
        FAIL() << "Expected throw";
    }
    catch(std::runtime_error const & err)
    {
    }
    catch(...)
    {
        FAIL() << "Expected throw";
    }
}

void UTGaudi2nBuffAlloc::nBuffAllocTester(std::function<uint16_t()> func)
{
    AllocW allocator;

    constexpr uint16_t NUM_BUFF = NBuffAllocator::NUM_BUFF;

    uint64_t longSo = 1;

    std::array<uint64_t, NUM_BUFF> arr {};

    for (int i = 0; i < 10000; i++)
    {
        uint16_t n = func(); //(rand() % NUM_BUFF) + 1;

        AllocW::AllocRtn rtn = allocator.alloc(n  * CHUNK_SIZE);
        // check it is consecutive
        ASSERT_LE((rtn.offset / CHUNK_SIZE) + n, NUM_BUFF);

        // find the expected longSo
        uint64_t max = 0;
        for (uint16_t j = 0; j < n; j++)
        {
            max = std::max(max, arr[rtn.offset / CHUNK_SIZE + j]);
        }
        ASSERT_EQ(max, rtn.longSo);

        // set longSo
        allocator.setLongSo(longSo);
        ASSERT_EQ(longSo, allocator.getLastLongSo());
        // set in my datebase the values
        for (uint16_t j = 0; j < n; j++)
        {
            arr[rtn.offset / CHUNK_SIZE + j] = longSo;
        }
        longSo += 3; // just that it is not 1
    } // for(i)
}

TEST_F(UTGaudi2nBuffAlloc, nBuffAllocRandomBig)
{
    constexpr uint16_t NUM_BUFF = NBuffAllocator::NUM_BUFF;

    nBuffAllocTester([]() {return (rand() % NUM_BUFF) + 1; }); // allocate 1 - NUM_BUFF
}

TEST_F(UTGaudi2nBuffAlloc, nBuffAlloc1)
{
    nBuffAllocTester([]() { return 1; }); // allocate always 1
}

TEST_F(UTGaudi2nBuffAlloc, nBuffAllocRandom)
{
    constexpr uint16_t NUM_BUFF = NBuffAllocator::NUM_BUFF;

    nBuffAllocTester([]() {return (rand() % (NUM_BUFF / 2)) + 1; }); // allocate 1 - NUM_BUFF/2
}
