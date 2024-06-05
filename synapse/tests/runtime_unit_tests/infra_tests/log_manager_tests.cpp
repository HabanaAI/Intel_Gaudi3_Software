#include <gtest/gtest.h>
#include <fstream>
#include "infra/containers/slot_map.hpp"
#include "infra/log_manager.h"
#include "containers/perf_test.hpp"
#include "syn_logging.h"
using namespace synapse;
TEST(LogManagerTest, init)
{
    LogManager::instance();
}
//#define ENABLE_PERFORMANCE_TEST 1
TEST(LogManagerTest, periodic)
{
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);
    const int cnt = 100;
    for (int i = 0; i < cnt; ++i)
    {
        if (i > 0 && i < 10)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        LOG_PERIODIC_DEBUG(SYN_API, std::chrono::milliseconds(10), 10, "hello {}", i);
        if (i % 20 == 0 && i > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

TEST(LogManagerTest, TestLogOffPerf)
{
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 4);
    const int cnt   = 1000000;
    auto      start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cnt; ++i)
    {
        LOG_TRACE(SYN_API, "hello {}", 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / double(cnt) << "ns\n";
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);
}

/*
 * test crash handlers
 * can be tested only manually and one test at a time
 * each test crashes and in the log file it should add the log message from the test
 * */
int recursion(int i)
{
    int* v = (int*)alloca(100 * 1024);
    v[0]   = i * i;
    v[1]   = i * i + 1;
    if (i > 1)
    {
        return recursion(i - 1) + v[i & 1];
    }
    return 1;
}

static int findStringInLog(const std::string& strToFind, const char* filename = SYNAPSE_LOG_SINK_FILE_RT)
{
    const int searchRegion = 10000;  // search within [EndOfFile - searchRegion, EndOfFile]

    std::string logsFolder;
    LogManager::instance().getLogsFolderPath(logsFolder);
    std::ifstream ifs(logsFolder + "/" + filename);

    ifs.seekg(0, std::ios_base::end);
    if (ifs.tellg() > searchRegion)
    {
        ifs.seekg(-searchRegion, std::ios_base::end);
    }
    else
    {
        ifs.seekg(0, std::ios_base::beg);
    }
    std::string str;
    int occurrenceCount = 0;
    while (ifs.good() && !ifs.eof())
    {
        std::getline(ifs, str);
        if (str.find(strToFind) != std::string::npos)
        {
            occurrenceCount++;
        }
    }
    return occurrenceCount;
}

TEST(LogManagerTest, ForkFlushing)
{
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);
    LogManager::instance().enablePeriodicFlush(false);
    std::string logStr = "before fork " + std::to_string(time(nullptr));
    LOG_TRACE_T(SYN_API, "{}", logStr);
    int pid = fork();
    LogManager::instance().flush();
    if (pid == 0)
    {
        exit(0);
    }
    // make sure child process finished - wait a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_EQ(findStringInLog(logStr), 1) << "only one log string must be in the log";
}

TEST(LogManagerTest, logsFolderISAbsolute)
{
    std::string logsFolder;
    ASSERT_TRUE(LogManager::instance().getLogsFolderPath(logsFolder));
    ASSERT_TRUE(!logsFolder.empty());
    ASSERT_EQ(logsFolder[0], '/');
}

TEST(LogManagerTest, StackOverflowCrash)
{
    std::string logStr = "before stack overflow " + std::to_string(time(nullptr));
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);

    auto callable = [logStr]() {
        LOG_TRACE(SYN_API, "{}", logStr);
        recursion(100);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringInLog(logStr)) << "log string not found in the log";
}

// different logger
TEST(LogManagerTest, StackOverflowCrash_SYN_STREAM)
{
    std::string logStr = "before stack overflow SYN_STREAM " + std::to_string(time(nullptr));
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_STREAM, 0);

    auto callable = [logStr]() {
        LOG_TRACE(SYN_STREAM, "{}", logStr);
        recursion(100);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringInLog(logStr)) << "log string not found in the log";
}

// different file
TEST(LogManagerTest, StackOverflowCrash_GC)
{
    std::string logStr = "before stack overflow GC " + std::to_string(time(nullptr));
    LogManager::instance().set_log_level(synapse::LogManager::LogType::GC, 0);

    auto callable = [logStr]() {
        LOG_TRACE(GC, "{}", logStr);
        recursion(100);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringInLog(logStr, SYNAPSE_LOG_SINK_FILE) != 0) << "log string not found in the log";
}

void writeToPtr(int* ptr)
{
    *ptr = 55;
}

TEST(LogManagerTest, NullPtrCrash)
{
    std::string logStr = "before null ptr " + std::to_string(time(nullptr));
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);
    int* ptr = nullptr;

    auto callable = [logStr, ptr]() {
        LOG_TRACE(SYN_API, "{}", logStr);
        writeToPtr(ptr);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringInLog(logStr)) << "log string not found in the log";
}

TEST(LogManagerTest, IllegalPtrCrash)
{
    int* ptr = (int*)0xFFFFFFFFFFFFFFFF;
    std::string logStr = "before illegal ptr " + std::to_string((uint64_t)ptr) + " " + std::to_string(time(nullptr));
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);

    auto callable = [logStr, ptr]() {
        LOG_TRACE(SYN_API, "{}", logStr);
        writeToPtr(ptr);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringInLog(logStr)) << "log string not found in the log";
}

void throwException()
{
    throw 1;
}

TEST(LogManagerTest, Terminate)
{
    std::string logStr = "before terminate " + std::to_string(time(nullptr));
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);

    auto callable = [logStr]() {
        LOG_TRACE(SYN_API, "{}", logStr);
        struct S
        {
            ~S() { ::throwException(); }
        };
        S s;
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringInLog(logStr)) << "log string not found in the log";
}

TEST(LogManagerTest, LongTimeAfterLog)
{

    std::string logStr = "before long delay " + std::to_string(time(nullptr));
    LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 0);
    LogManager::instance().enablePeriodicFlush(false);

    LOG_TRACE(SYN_API, "{}", logStr);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ASSERT_FALSE(findStringInLog(logStr)) << "log string not found in the log";
    LogManager::instance().enablePeriodicFlush();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ASSERT_TRUE(findStringInLog(logStr)) << "log string not found in the log";

}

[[maybe_unused]] static OperationFullMeasurementResults testLongMessageTrace(TestParams params)
{
    unsigned i   = 0;
    auto     res = measure(
        [&i](unsigned thread_id) {
            LOG_TRACE(SYN_API,
                      "{}: Patching (DC {} PP-Index in Stage {:x}) DC-Address 0x{:x} Offset-in-DC 0x{:x},"
                      " section {}, effective-address 0x{:x} patched-location 0x{:x} patch-type {} value 0x{:x}",
                      HLLOG_FUNC,
                      i,
                      i + 1,
                      i + 2,
                      i + 3,
                      i + 4,
                      //                pp.node_exe_index,
                      i + 10,
                      i + 20,
                      i + 30,
                      i + 40);
            ++i;
        },
        "long msg buffer compile trace",
        params);

    return res;
}

[[maybe_unused]] static OperationFullMeasurementResults testMediumMessageTrace(TestParams params)
{
    unsigned i   = 0;
    auto     res = measure(
        [&i](unsigned thread_id) {
            LOG_TRACE(SYN_API,
                      "{} section {:x} calc {:x} expected {:x} valid hi/low {:x}/{:x}",
                      HLLOG_FUNC,
                      i,
                      i + 2,
                      i + 5,
                      i + 10,
                      1 + 20);
            ++i;
        },
        "medium msg buffer trace",
        params);

    return res;
}

[[maybe_unused]] static OperationFullMeasurementResults testShortMessageTrace(TestParams params)
{
    unsigned i   = 0;
    auto     res = measure(
        [&i](unsigned thread_id) {
            LOG_TRACE(SYN_API, "DW HIGH: Current val {:x} different from previous {:x}", i, i + 2);
            ++i;
        },
        "short msg buffer trace",
        params);

    return res;
}

#if ENABLE_PERFORMANCE_TEST
TEST(LogManagerTest, logger_performance)
{
    TestParams params;
    params.maxNbThreads            = 1;
    params.minNbThreads            = 1;
    params.nbTests                 = 100;
    params.internalLoopInterations = 10000;
    params.interations_per_sleep   = 0;
    PrintTestResults("logger_performance",
                     {testLongMessageTrace(params), testMediumMessageTrace(params), testShortMessageTrace(params)});
}
#endif
