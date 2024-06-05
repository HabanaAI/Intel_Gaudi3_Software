#include <gtest/gtest.h>
#include <hl_logger/impl/recent_elements_queue.hpp>
#include "run_test_mt.hpp"

static void fillItems(containers::ConcurrentRecentElementsQueue<int>& items, int cnt)
{
    for (int i = 0; i < cnt / 2; ++i)
    {
        items.push(i);
    }
    for (int i = cnt / 2; i < cnt; ++i)
    {
        items.emplace(i);
    }
}

static void verifyItems(containers::ConcurrentRecentElementsQueue<int>& items,
                        size_t                                                          cnt,
                        containers::ElementsOrder order = containers::ElementsOrder::chronological)
{
    std::vector<int> vals;
    items.processAndClear([&](int& v) { vals.push_back(v); }, order);
    int offset = 0;
    if (cnt < items.capacity())
    {
        ASSERT_EQ(vals.size(), cnt);
    }
    else
    {
        ASSERT_EQ(vals.size(), items.capacity());
        offset = cnt - items.capacity();
    }

    if (order == containers::ElementsOrder::reverseChronological)
    {
        std::reverse(vals.begin(), vals.end());
    }

    for (unsigned i = 0; i < vals.size(); ++i)
    {
        ASSERT_EQ(vals[i], i + offset);
    }
}

TEST(RecentElementsQueue, basic_ops)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, 100);
    verifyItems(items, 100);
}

TEST(RecentElementsQueue, clear)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, 100);
    items.clear();
    verifyItems(items, 0);
}

TEST(RecentElementsQueue, basic_ops_rev)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, 100);
    verifyItems(items, 100, containers::ElementsOrder::reverseChronological);
}

TEST(RecentElementsQueue, basic_ops_capacity)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, items.capacity());
    verifyItems(items, items.capacity());
}

TEST(RecentElementsQueue, basic_ops_capacity_plus_1)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, items.capacity() + 1);
    verifyItems(items, items.capacity() + 1);
}

TEST(RecentElementsQueue, basic_ops_capacity_plus_1_rev)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, items.capacity() + 1);
    verifyItems(items, items.capacity() + 1, containers::ElementsOrder::reverseChronological);
}

TEST(RecentElementsQueue, basic_ops_capacity_2x)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, items.capacity() * 2);
    verifyItems(items, items.capacity() * 2);
}

TEST(RecentElementsQueue, basic_ops_capacity_2x_rev)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, items.capacity() * 2);
    verifyItems(items, items.capacity() * 2, containers::ElementsOrder::reverseChronological);
}

TEST(RecentElementsQueue, basic_ops_capacity_2_5x)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, (items.capacity() * 5) / 2);
    verifyItems(items, (items.capacity() * 5) / 2);
}

TEST(RecentElementsQueue, basic_ops_capacity_2_5x_rev)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    fillItems(items, (items.capacity() * 5) / 2);
    verifyItems(items, (items.capacity() * 5) / 2, containers::ElementsOrder::reverseChronological);
}

TEST(RecentElementsQueue, fill_process_loop)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    for (unsigned i = 0; i < 10; ++i)
    {
        fillItems(items, (items.capacity() * (i + 1)) / 2);
        verifyItems(items, (items.capacity() * (i + 1)) / 2);
    }
}

TEST(RecentElementsQueue, basic_ops_MultiThread)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    runTestsMT([&](unsigned, unsigned) { fillItems(items, (items.capacity() * 5) / 2); }, 200, 6);
}

TEST(RecentElementsQueue, basic_ops_MultiThread_full)
{
    containers::ConcurrentRecentElementsQueue<int> items(1024);
    std::vector<std::thread>           threads;
    const unsigned                     nbThreads = 8;
    std::atomic<bool>                  stop {false};
    for (unsigned i = 0; i < nbThreads; ++i)
    {
        threads.push_back(std::thread(
            [&](unsigned thread_id) {
                unsigned v = 0;
                while (!stop.load(std::memory_order_relaxed))
                {
                    items.push(v);
                    ++v;
                }
            },
            i));
    };

    for (unsigned i = 0; i < 3; ++i)
    {
        threads.push_back(std::thread(
            [&](unsigned thread_id) {
                while (!stop.load(std::memory_order_relaxed))
                {
                    std::vector<unsigned> vals;
                    if (thread_id == 0)
                    {
                        items.clear();
                    }
                    else
                    {
                        items.processAndClear([&vals](auto v) { vals.push_back(v); });
                    };
                    std::this_thread::sleep_for(std::chrono::milliseconds((thread_id + 1) * 200));
                }
            },
            i));
    };
    std::this_thread::sleep_for(std::chrono::seconds(3));
    stop = true;
    for (auto& t : threads)
    {
        t.join();
    }
}

void testItemOverwrite()
{
    containers::ConcurrentRecentElementsQueue<int> items(8);

    std::vector<std::thread> threads;
    std::atomic<bool>        stop {false};
    // write threads
    for (unsigned i = 0; i < 4; ++i)
    {
        threads.push_back(std::thread(
            [&](unsigned thread_id) {
                unsigned v = 0;
                while (!stop.load(std::memory_order_relaxed))
                {
                    items.push(v);
                    ++v;
                }
            },
            i));
    };
    int mismatchCount = 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    items.processAndClear([&](int& v, bool epochMatch) {
        if (!epochMatch) mismatchCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    stop = true;
    for (auto& t : threads)
    {
        t.join();
    }

    ASSERT_NE(mismatchCount, 0);
}
TEST(RecentElementsQueue, processItemsWithOvewrite)
{
    testItemOverwrite();
    testItemOverwrite();
    testItemOverwrite();
    testItemOverwrite();
}