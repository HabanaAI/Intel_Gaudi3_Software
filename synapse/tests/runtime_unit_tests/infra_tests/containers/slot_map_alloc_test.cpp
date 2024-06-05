#include <gtest/gtest.h>
#include "infra/containers/slot_map_alloc.hpp"
#include "run_test_mt.hpp"

void basicOperationsRefCounting(ConcurrentSlotMapAlloc<int>& itemsMap)
{
    auto newItem1 = itemsMap.insert(0, 55);
    auto newItem2 = itemsMap.insert(0, 56);

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
    auto newItem3 = itemsMap.insert(0, 55);
    ASSERT_NE(newItem3.second, nullptr);
    // handles of reused item and of the deleted one must be different
    ASSERT_NE(newItem3.first, newItem1.first);
    newItem3.second.reset();
    ASSERT_TRUE(itemsMap.erase(newItem3.first));
}
TEST(SlotMapAllocTest, basic_ops_RefCounting)
{
    ConcurrentSlotMapAlloc<int> itemsMap;
    basicOperationsRefCounting(itemsMap);
}

TEST(SlotMapAllocTest, basic_ops_RefCounting_MutipleThraeds)
{
    ConcurrentSlotMapAlloc<int> itemsMap;
    runTestsMT([&itemsMap](unsigned, unsigned) { basicOperationsRefCounting(itemsMap); }, 1000, 6);
}

TEST(SlotMapAllocTest, constTest)
{
    ConcurrentSlotMapAlloc<int> itemsMap;
    auto [handle, ptr] = itemsMap.insert(0, 123);
    *ptr += 10;
    const ConcurrentSlotMapAlloc<int> & itemsMapConst = itemsMap;
    auto ptr2 = itemsMapConst[handle];
    auto ptr3 = ptr2;
    static_assert(std::is_const_v<decltype(ptr2)::element_type>);
}

void basicOperationsNoRefCounting(ConcurrentSlotMapAlloc<int>& itemsMap)
{
    auto newItem1 = itemsMap.insert(0, 55);
    auto newItem2 = itemsMap.insert(0, 56);

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
    // succeed without ref counting
    ASSERT_TRUE(itemsMap.erase(newItem1.first));
    newItem1.second.reset();
    item1.reset();
    // re-deletion
    ASSERT_FALSE(itemsMap.erase(newItem1.first));

    // access by wrong handle
    auto itemWrong = itemsMap[wrongHandle];
    ASSERT_EQ(itemWrong, nullptr);
    auto itemWrong2 = itemsMap[newItem1.first];
    ASSERT_EQ(itemWrong2, nullptr);

    // check re-use of deleted items
    auto newItem3 = itemsMap.insert(0, 55);
    // handles of reused item and of the deleted one must be different
    ASSERT_NE(newItem3.first, newItem1.first);
}
TEST(SlotMapAllocTest, basic_ops_noRefCount)
{
    ConcurrentSlotMapAlloc<int> itemsMap(1000, SlotMapChecks::noRefCounting);
    basicOperationsNoRefCounting(itemsMap);
}

TEST(SlotMapAllocTest, basic_ops_noRefCount_MultipleThreads)
{
    ConcurrentSlotMapAlloc<int> itemsMap(10000, SlotMapChecks::noRefCounting);
    runTestsMT([&itemsMap](unsigned, unsigned) { basicOperationsNoRefCounting(itemsMap); }, 500, 6);
}

namespace
{
struct CountInstances
{
    CountInstances() { ctors++; }
    ~CountInstances() { dtors++; }
    static void reset()
    {
        ctors = 0;
        dtors = 0;
    }
    static std::atomic<unsigned> ctors;
    static std::atomic<unsigned> dtors;
};

std::atomic<unsigned> CountInstances::ctors {0};
std::atomic<unsigned> CountInstances::dtors {0};
}  // namespace

TEST(SlotMapAllocTest, creationCheck)
{
    CountInstances::reset();
    {
        ConcurrentSlotMapAlloc<CountInstances> itemsMap(2);
        ASSERT_EQ(CountInstances::ctors.load(), 0);
        SMHandle h1, h2;
        {
            h1 = itemsMap.insert(0).first;
            h2 = itemsMap.insert(0).first;
            ASSERT_EQ(CountInstances::ctors.load(), 2);
            ASSERT_EQ(CountInstances::dtors.load(), 0);
        }
        ASSERT_TRUE(itemsMap.erase(h1));
        ASSERT_TRUE(itemsMap.erase(h2));
        ASSERT_EQ(CountInstances::dtors.load(), 2);

        {
            h1 = itemsMap.insert(0).first;
            h2 = itemsMap.insert(0).first;
            ASSERT_EQ(CountInstances::ctors.load(), 4);
            ASSERT_EQ(CountInstances::dtors.load(), 2);
        }
    }
    ASSERT_EQ(CountInstances::dtors.load(), 4);
}

TEST(SlotMapAllocTest, uninitFunc)
{
    unsigned uninitCalls = 0;

    ConcurrentSlotMapAlloc<int, 1024> itemsMap(100, SlotMapChecks::full, [&](int*) {
        uninitCalls++;
        return true;
    });
    ASSERT_EQ(uninitCalls, 0);
    auto handle1 = itemsMap.insert(0, 55).first;
    ASSERT_EQ(uninitCalls, 0);
    auto handle2 = itemsMap.insert(0, 56).first;
    ASSERT_EQ(uninitCalls, 0);

    ASSERT_TRUE(itemsMap.erase(handle1));
    ASSERT_TRUE(itemsMap.erase(handle2));

    ASSERT_EQ(uninitCalls, 2);

    itemsMap.insert(0, 55);
    itemsMap.insert(0, 56);
    itemsMap.insert(0, 56);
    itemsMap.eraseAll();

    ASSERT_EQ(uninitCalls, 5);
}

TEST(SlotMapAllocTest, concurrentInsertDelete)
{
    const unsigned              maxHandlesCount = 100000;
    ConcurrentSlotMapAlloc<int> itemsMap;
    const unsigned              nbThreads = 6;

    std::array<std::atomic<SMHandle>, maxHandlesCount * nbThreads> handles;
    for (auto& hndl : handles)
    {
        hndl.store(0);
    }
    runTestsMT(
        [&](unsigned thread_id, unsigned i) {
            unsigned idx           = i + thread_id * maxHandlesCount;
            SMHandle newItemHandle = 0;
            {
                auto newItem = itemsMap.insert(idx);
                ASSERT_NE(newItem.second.get(), nullptr)
                    << "failed to create: thd:" << thread_id << " i: " << i << " idx: " << idx;
                newItemHandle = newItem.first;
            }
            handles[idx] = newItemHandle;
            if (i > 1000)
            {
                unsigned ersIdx = i - 1000 + ((thread_id + 1) % nbThreads) * maxHandlesCount;
                while (handles[ersIdx] == 0)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                {
                    auto item = itemsMap[handles[ersIdx]];
                    ASSERT_NE(item.get(), nullptr)
                        << "failed to access: thd:" << thread_id << " i: " << i << " idx: " << ersIdx;
                    bool success = itemsMap.erase(handles[ersIdx]);
                    ASSERT_FALSE(success)
                        << "succeed to erase BUT must fail: thd:" << thread_id << " i: " << i << " idx: " << ersIdx;
                }
                bool success = itemsMap.erase(handles[ersIdx]);
                ASSERT_TRUE(success) << "failed to erase BUT must succeed: thd:" << thread_id << " i: " << i
                                     << " idx: " << ersIdx;
            }
        },
        maxHandlesCount,
        nbThreads);
}

TEST(SlotMapAllocTest, concurrentEraseOfTheSameItem)
{
    static const unsigned nbThreads = 6;

    static ConcurrentSlotMapAlloc<int> itemsMap(10);
    static std::atomic<SMHandle>       handle;
    static std::atomic<int>            successfulDeletions {0};
    static std::atomic<int>            unsuccessfulDeletions {0};

    static std::condition_variable cv;
    static std::atomic<bool>       go {false};
    for (unsigned i = 0; i < 1000; ++i)
    {
        handle = itemsMap.insert(0).first;
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