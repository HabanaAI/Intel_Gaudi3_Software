#include <gtest/gtest.h>
#include <hl_logger/impl/recent_elements_queue.hpp>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "perf_test.hpp"
#include <atomic>

struct S
{
    std::atomic<int> a;
    char             b[60];
};

static OperationFullMeasurementResults testOneAtomic(TestParams params)
{
    static S s[20] {};
    return measure([](unsigned thread_id) { s[0].a += 76; }, "one value atomic", params);
}

static OperationFullMeasurementResults testHistoryBuffer(TestParams params)
{
    containers::ConcurrentRecentElementsQueue<unsigned> buffer(1024);

    return measure([&](unsigned thread_id) { buffer.push(2); }, "HistoryBuffer unsigned", params);
}

TEST(RecentElementsQueue, DISABLED_basic_ops_perf_test)
{
    TestParams params;
    params.maxNbThreads            = 10;
    params.minNbThreads            = 1;
    params.nbTests                 = 200;
    params.internalLoopInterations = 2000;
    params.interations_per_sleep   = 0;
    PrintTestResults("History buffer performance check", {testOneAtomic(params), testHistoryBuffer(params)});
}