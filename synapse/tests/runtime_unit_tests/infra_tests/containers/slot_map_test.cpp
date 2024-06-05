#include <gtest/gtest.h>
#include "infra/containers/slot_map.hpp"
#include "run_test_mt.hpp"
static void BasicOperations(ConcurrentSlotMap<int>& itemsMap)
{
    int  v1       = 55;
    int  v2       = 56;
    auto newItem1 = itemsMap.insert(&v1, 0);
    auto newItem2 = itemsMap.insert(&v2, 0);

    ASSERT_NE(newItem1.second, nullptr);
    ASSERT_NE(newItem2.second, nullptr);

    ASSERT_EQ(*newItem1.second, 55);
    ASSERT_EQ(*newItem2.second, 56);

    ASSERT_NE(newItem1.first, newItem2.first);

    // access by handle
    auto item1 = itemsMap[newItem1.first];
    ASSERT_NE(item1, nullptr);
    ASSERT_EQ(*item1, 55);
    auto item2 = itemsMap[newItem2.first];
    ASSERT_NE(item2, nullptr);
    ASSERT_EQ(*item2, 56);

    // erase wrong handle
    const uint64_t wrongHandle = 0;
    ASSERT_FALSE(itemsMap.erase(wrongHandle));
    // erase while in use
    ASSERT_FALSE(itemsMap.erase(newItem1.first));
    newItem1.second.reset();
    item1.reset();
    // now not in use
    ASSERT_TRUE(itemsMap.erase(newItem1.first));
    // re-deletion
    ASSERT_FALSE(itemsMap.erase(newItem1.first));

    // access by wrong handle
    auto itemWrong = itemsMap[wrongHandle];
    ASSERT_EQ(itemWrong, nullptr);
    auto itemWrong2 = itemsMap[newItem1.first];
    ASSERT_EQ(itemWrong2, nullptr);

    // check re-use of deleted items
    auto newItem3 = itemsMap.insert(&v1, 0);
    ASSERT_EQ(newItem3.second.get(), &v1);
    // handles of reused item and aof the deleted one must be different
    ASSERT_NE(newItem3.first, newItem1.first);
}
TEST(SlotMapTest, basic_ops)
{
    ConcurrentSlotMap<int> itemsMap;
    BasicOperations(itemsMap);
}

TEST(SlotMapTest, basic_ops_MultiThread)
{
    ConcurrentSlotMap<int> itemsMap;
    runTestsMT([&](unsigned, unsigned) { BasicOperations(itemsMap); }, 1000, 6);
}

TEST(SlotMapTest, concurrentEraseOfTheSameItem)
{
    static const unsigned nbThreads = 6;

    static ConcurrentSlotMap<int>  itemsMap(10);
    static volatile SMHandle       handle;
    static std::atomic<int>        successfulDeletions {0};
    static std::atomic<int>        unsuccessfulDeletions {0};
    int                            value = 0;
    static std::condition_variable cv;
    static std::atomic<bool>       go {false};
    for (unsigned i = 0; i < 1000; ++i)
    {
        handle = itemsMap.insert(&value, 0).first;
        go     = false;
        std::vector<std::thread> threads;
        successfulDeletions   = 0;
        unsuccessfulDeletions = 0;
        for (unsigned t = 0; t < nbThreads; ++t)
        {
            threads.push_back(std::thread([]() {
                while (!go)
                    ;
                if (itemsMap.erase(handle)) ++successfulDeletions;
                else
                    ++unsuccessfulDeletions;
            }));
        };
        go = true;
        for (unsigned t = 0; t < nbThreads; ++t)
        {
            threads[t].join();
        }
        ASSERT_EQ(successfulDeletions.load(), 1);
        ASSERT_EQ(unsuccessfulDeletions.load(), nbThreads - 1);
    }
}

TEST(SlotMapTest, nonZeroHandle)
{
    ConcurrentSlotMap<int> itemsMap;
    int                    v = 1;
    for (unsigned i = 0; i < 10000000U; ++i)
    {
        if (i == 0x100000)
        {
            i = 0x100000;
        }
        auto newItem = itemsMap.insert(&v, 0);
        if (newItem.first == 0)
        {
            std::cerr << "failed. i: " << i << std::endl;
        }
        ASSERT_NE(newItem.first, 0);
        newItem.second.reset();
        itemsMap.erase(newItem.first);
    }
}

TEST(SlotMapTest, constTest)
{
    ConcurrentSlotMap<int> itemsMap;
    int v = 0;
    auto [handle, ptr] = itemsMap.insert(&v, 0);
    *ptr += 10;
    const ConcurrentSlotMap<int> & itemsMapConst = itemsMap;
    auto ptr2 = itemsMapConst[handle];
    auto ptr3 = ptr2;
    static_assert(std::is_const_v<decltype(ptr2)::element_type>);
}