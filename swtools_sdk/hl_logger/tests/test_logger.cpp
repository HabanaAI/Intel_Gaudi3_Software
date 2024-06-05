#include <gtest/gtest.h>
#include <fstream>
#include "hl_logger/hllog_core.hpp"
#include "synapse_logger.h"
#include <thread>
#include "utils.h"
#include "perf_test.hpp"
#include <list>

#if !defined __GNUC__ || __GNUC__ >= 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif


enum class MyEnum
{
    EnumItem_1,
    EnumItem_2
};

/*
TEST(hl_logger_test, basic_mt)
{
    hl_logger::enablePeriodicFlush();
    bool enableFlush = true;
    SET_LOGGER_LEVEL(SCAL_1, HLLOG_LEVEL_TRACE);
    HLLOG_TRACE(SCAL_1, "hello {} {} {}", "world", "!", HLLOG_TO_STRING(HLLOG_INLINE_API_NAMESPACE));
    std::vector<std::thread> threads;
    for (unsigned t = 0 ; t < 5; ++t)
    threads.push_back(std::thread([t, &enableFlush]() {
        for (unsigned i = 0 ; i < 10000000; ++i) {
            HLLOG_TRACE(SYN_API, "hello {} {} {}", "world", "!", HLLOG_TO_STRING(HLLOG_INLINE_API_NAMESPACE));
            if (t == 0 && i % 1000 == 0) {
                hl_logger::enablePeriodicFlush(enableFlush);
                enableFlush = !enableFlush;
            }
        }
    }));
    for (auto & t : threads)
    {
        t.join();
    }
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    // on demand - log file is created on the first usage of the logger
    HLLOG_TRACE(SYN_API, "hello world_syn api");
}
*/

TEST(hl_logger_test, versionCheck)
{
    ASSERT_TRUE(!hl_logger::getVersion().commitSHA1.empty());
}

static constexpr char char_str[] = "some_char string";
TEST(hl_logger_test, basic)
{
    hl_logger::enablePeriodicFlush();
    std::cout << "Start :" << std::endl;
    HLLOG_ERR(SCAL_1, "hello world with disabled logger");
    struct A
    {
        int v : 2;
    };
    A a {3};
    HLLOG_TRACE(SCAL_1, "hello world with disabled logger {}", a.v);
    HLLOG_TRACE(SCAL_1, "{}", char_str);
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_1, HLLOG_LEVEL_TRACE);
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);
    HLLOG_SET_LOGGING_LEVEL(SCAL_1, 0);
    HLLOG_TRACE(SCAL_1, "hello {} {} {}", "world", "!", HLLOG_TO_STRING(HLLOG_INLINE_API_NAMESPACE));
    HLLOG_TRACE(SCAL_1, "test values {} {} XXX==={:<15}===XXXX", "one", a.v, MyEnum::EnumItem_2);
    HLLOG_TRACE(SCAL_1, "test values {} {:#x} XXX==={:#<15x}===XXXX", "one", a.v, MyEnum::EnumItem_2);
    HLLOG_TRACE(SCAL_1, "test values {} {:x} XXX==={:<15x}===XXXX", "one", a.v, MyEnum::EnumItem_2);

    std::vector<int> vals {1, 2, 3, 4, 5, 6, 7, 8, 9};
    HLLOG_TRACE(SYN_API, "hello {}", fmt::join(vals.begin(), vals.end(), ","));
    std::thread t([&]() {
        HLLOG_TRACE(SCAL_1, "hello {} {} {}", "world", "!", HLLOG_TO_STRING(HLLOG_INLINE_API_NAMESPACE));
        HLLOG_TRACE(SCAL_1, "test values {} {} {}", "one", a.v, MyEnum::EnumItem_2);
    });
    t.join();
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    // on demand - log file is created on the first usage of the logger
    HLLOG_TRACE(SYN_API, "hello world_syn api");
    int v = 0;
    HLLOG_TRACE(SYN_API, "address: {}", &v);
    HLLOG_TRACE(SYN_API, "address: {:p}", &v);

}

TEST(hl_logger_test, basic2)
{
    hl_logger::enablePeriodicFlush();
    std::cout << "Start :" << std::endl;
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_API, HLLOG_LEVEL_TRACE);
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_API_3, HLLOG_LEVEL_TRACE);

    HLLOG_TRACE(SCAL_API, "hello with time");
    HLLOG_TRACE(SCAL_API_3, "hello without time in create but with time in reality");
}

TEST(hl_logger_test, lazyLogs)
{
    hl_logger::enablePeriodicFlush();
    {
        HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_TRACE);
        hl_logger::enablePeriodicFlush();
        std::cout << "Start :" << std::endl;
        HLLOG_ERR(SCAL_1, "hello world with disabled logger");
        struct A {
            int v: 2;
        };
        A a{3};
        HLLOG_TRACE(SCAL_1, "hello world with disabled logger {}", a.v);
        HLLOG_TRACE(SCAL_1, "{}", char_str);

        hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_1, HLLOG_LEVEL_TRACE);
        hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);
        HLLOG_SET_LOGGING_LEVEL(SCAL_1, 0);
        HLLOG_TRACE(SCAL_1, "hello {} {} {}", "world", "!", HLLOG_TO_STRING(HLLOG_INLINE_API_NAMESPACE));
        HLLOG_TRACE(SCAL_1, "test values {} {} XXX==={:<15}===XXXX", "one", a.v, MyEnum::EnumItem_2);
        HLLOG_TRACE(SCAL_1, "test values {} {:#x} XXX==={:#<15x}===XXXX", "one", a.v, MyEnum::EnumItem_2);
        HLLOG_TRACE(SCAL_1, "test values {} {:x} XXX==={:<15x}===XXXX", "one", a.v, MyEnum::EnumItem_2);

        const std::vector<int> vals{1, 2, 3, 4, 5, 6, 7, 8, 9};
        const std::vector<int> & vals2 = vals;
        HLLOG_TRACE(SYN_API, "hello {}", fmt::join(vals.begin(), vals.end(), ","));
        HLLOG_TRACE(SYN_API, "hello {}", fmt::join(vals2.begin(), vals2.end(), ","));
        const std::array<int, 9> vals_{1, 2, 3, 4, 5, 6, 7, 8, 9};
        const std::array<int, 9> & vals2_ = vals_;
        HLLOG_TRACE(SYN_API, "hello {}", fmt::join(vals_.begin(), vals_.end(), ","));
        HLLOG_TRACE(SYN_API, "hello {}", fmt::join(vals2_.begin(), vals2_.end(), ","));

        std::list<int> vals_l;
        vals_l.push_back(1);
        vals_l.push_back(2);
        vals_l.push_back(3);
        vals_l.push_back(4);
        vals_l.push_back(5);
        vals_l.push_back(6);
        vals_l.push_back(7);
        const std::list<int> & vals2_l = vals_l;
        HLLOG_TRACE(SYN_API, "hello {}", fmt::join(vals_l.begin(), vals_l.end(), ","));
        HLLOG_TRACE(SYN_API, "hello {}", fmt::join(vals2_l.begin(), vals2_l.end(), ","));

        std::atomic<int> atomic_v = 1;
        HLLOG_TRACE(SYN_API, "atomic {}", atomic_v);

        HLLOG_TRACE(SYN_API, "hello str {}", std::string("a long string long long long"));
        HLLOG_TRACE(SYN_API, "hello temp str {}", std::string("a long string long long long").c_str());
        std::thread t([&]() {
            HLLOG_TRACE(SCAL_1, "hello {} {} {}", "world", "!", HLLOG_TO_STRING(HLLOG_INLINE_API_NAMESPACE));
            HLLOG_TRACE(SCAL_1, "test values {} {} {}", "one", a.v, MyEnum::EnumItem_2);
        });
        t.join();
        // std::this_thread::sleep_for(std::chrono::seconds(5));
        // on demand - log file is created on the first usage of the logger
        HLLOG_TRACE(SYN_API, "hello world_syn api");
        int v = 0;
        HLLOG_TRACE(SYN_API, "address: {}", &v);
        HLLOG_TRACE(SYN_API, "address: {:p}", &v);
        HLLOG_TRACE(SYN_API,"{}: fd={} Failed to load config from {}. config not found", __FUNCTION__, 1, std::string("str"));
        HLLOG_ERR(SYN_API, "a string with no braces");
    }
    hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LoggerTypes::DFA));
}

TEST(hl_logger_test, lazyLogsTemps)
{
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_TRACE);
    char buffer[100] = "hello world";
    HLLOG_TRACE(SYN_API, "temp: {}", buffer);
    buffer[0] = 'X';
    buffer[1] = 0;
    hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LoggerTypes::DFA));
}

class HclImmediateData
{
    // fmt support
    friend std::ostream& operator<<(std::ostream& os, const HclImmediateData& id){ return os << "HclImmediateData: " << id.raw_ << " " << id.s_;};

public:
    union
    {
        struct
        {
            const uint32_t data_ : 16;
            const uint32_t remoteRank_ : 10;
            const uint32_t comm_ : 4;
            const uint32_t isArt_ : 1;
            const uint32_t isHclTag_ : 1;
        } __attribute__((packed));

        const uint32_t raw_;
    };
    std::string s_;
    HclImmediateData(uint32_t raw) : raw_(raw){s_ = "xxxxxxxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyy raw value is: " + std::to_string(raw);};
    HclImmediateData(bool isArt, int rank, int comm, uint16_t data) : raw_{}{};
    HclImmediateData(HclImmediateData&&) = default;
    HclImmediateData(HclImmediateData const &) = delete;
    explicit operator uint32_t() const { return raw_; }
};

TEST(hl_logger_test, ostreamTest)
{
    std::string s = fmt::format("{}", HclImmediateData(1));
    HLLOG_TRACE(SYN_API, "1: {}", s);
    HLLOG_TRACE(SYN_API, "2: {}", HclImmediateData(1));

}

TEST(hl_logger_test, recreateLogger)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);
    HLLOG_TRACE(SYN_API, "hello recreateLogger");
    HLLOG_DROP(SYN_API);
    HLLOG_TRACE(SYN_API, "hello22 recreateLogger");
}

TEST(hl_logger_test, emptyLogger)
{
    //should not crash
    hl_logger::log(nullptr, HLLOG_LEVEL_ERROR, "Empty Logger msg");
}

TEST(hl_logger_test, createLogger)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_TEST, HLLOG_LEVEL_TRACE);
    std::string filename = "syn_test.log";
    std::remove((hl_logger::getLogsFolderPath() + "/" + filename).c_str());
    const std::string str0 = "hello syn_test 0";
    const std::string str1 = "hello syn_test 1";
    const std::string str2 = "hello syn_test 2";
    const std::string str3 = "hello syn_test 3";
    const std::string str4 = "hello syn_test 4";
    HLLOG_TRACE(SYN_TEST, "{}", str0); // no output
    HLLOG_DROP(SYN_TEST);
    hl_logger::LoggerCreateParams params;
    params.logFileName = filename;
    params.defaultLoggingLevel = hl_logger::defaultLoggingLevel;

    hl_logger::createLogger(synapse::LoggerTypes::SYN_TEST, params);
    HLLOG_TRACE(SYN_TEST, "{}", str1); // should work
    HLLOG_DROP(SYN_TEST);
    HLLOG_TRACE(SYN_TEST, "{}", str2); // should not work
    hl_logger::createLogger(synapse::LoggerTypes::SYN_TEST, params);
    HLLOG_TRACE(SYN_TEST, "{}", str3); // should work
    HLLOG_DROP(SYN_TEST);
    params.defaultLoggingLevel = HLLOG_LEVEL_ERROR;
    hl_logger::createLogger(synapse::LoggerTypes::SYN_TEST, params);
    HLLOG_TRACE(SYN_TEST, "{}", str4); // should work
    ASSERT_TRUE(findStringsCountInLog(str0, filename) == 0) << "log string found in the log but should not be";
    ASSERT_TRUE(findStringsCountInLog(str1, filename) == 1) << "log string not found in the log";
    ASSERT_TRUE(findStringsCountInLog(str2, filename) == 0) << "log string found in the log but should not be";
    ASSERT_TRUE(findStringsCountInLog(str3, filename) == 1) << "log string not found in the log";
    ASSERT_TRUE(findStringsCountInLog(str4, filename) == 0) << "log string found in the log but should not be";

}

TEST(hl_logger_test, additionalFileSink)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);
    auto curSinks = hl_logger::getSinks(synapse::LoggerTypes::SYN_API);
    std::string filename = "add_test.log";
    std::string fullFilename = hl_logger::getLogsFolderPath() + "/" + filename;
    std::remove(fullFilename.c_str());
    auto sinks = hl_logger::getSinks(synapse::LoggerTypes::SYN_API);
    hl_logger::addFileSink(synapse::LoggerTypes::SYN_API, filename, 1024*1024, 1);
    std::string line1 = "additionalFileSink line 1 " + std::to_string(time(nullptr));
    HLLOG_TRACE(SYN_API, "{}", line1);
    hl_logger::flush(synapse::LoggerTypes::SYN_API);
    ASSERT_TRUE(findStringsCountInLog(line1, filename) == 1);
    ASSERT_TRUE(findStringsCountInLog(line1, "syn_api_log.txt") == 1);
    hl_logger::setSinks(synapse::LoggerTypes::SYN_API, std::move(sinks));

    std::string line2 = "additionalFileSink line 2 " + std::to_string(time(nullptr));
    HLLOG_TRACE(SYN_API, "{}", line2);
    hl_logger::flush(synapse::LoggerTypes::SYN_API);
    ASSERT_TRUE(findStringsCountInLog(line2, filename) == 0);
    ASSERT_TRUE(findStringsCountInLog(line2, "syn_api_log.txt") == 1);
}

TEST(hl_logger_test, filenames)
{
    auto filenames = hl_logger::getSinksFilenames(synapse::LoggerTypes::SYN_API);
    ASSERT_EQ(filenames.size(), 1);
    ASSERT_NE(filenames[0].find("syn_api_log.txt"), std::string::npos);
    std::string filename = "add_test.log";
    std::string fullFilename = hl_logger::getLogsFolderPath() + "/" + filename;
    auto sinks = hl_logger::getSinks(synapse::LoggerTypes::SYN_API);
    hl_logger::addFileSink(synapse::LoggerTypes::SYN_API, filename, 1024*1024, 1);
    filenames = hl_logger::getSinksFilenames(synapse::LoggerTypes::SYN_API);
    ASSERT_EQ(filenames.size(), 2);
    ASSERT_NE(filenames[0].find("syn_api_log.txt"), std::string::npos);
    ASSERT_EQ(filenames[1].find(fullFilename), 0);
    hl_logger::setSinks(synapse::LoggerTypes::SYN_API, sinks);
    filenames = hl_logger::getSinksFilenames(synapse::LoggerTypes::SYN_API);
    ASSERT_EQ(filenames.size(), 1);
    ASSERT_NE(filenames[0].find("syn_api_log.txt"), std::string::npos);
}

TEST(hl_logger_test, filenames_after_changing_logs_folder)
{
    auto sinks_orig = hl_logger::getSinks(synapse::LoggerTypes::SYN_API);

    std::string newLogFilename = "add_test.log";
    std::string newLogFullpath = hl_logger::getLogsFolderPath() + "/" + newLogFilename;
    hl_logger::addFileSink(synapse::LoggerTypes::SYN_API, newLogFilename, 1024*1024, 1);

    auto filenames = hl_logger::getSinksFilenames(synapse::LoggerTypes::SYN_API);
    ASSERT_EQ(filenames.size(), 2);
    ASSERT_EQ(filenames[0], hl_logger::getLogsFolderPath() + "/syn_api_log.txt");
    ASSERT_EQ(filenames[1], newLogFullpath);

    const std::string new_logs_dir = hl_logger::getLogsFolderPath() + "/new-log-dir";
    // reinit existing file sinks to custom directory
    hl_logger::setLogsFolderPath(new_logs_dir);

    auto filenamesAfterReinit = hl_logger::getSinksFilenames(synapse::LoggerTypes::SYN_API);
    ASSERT_EQ(filenamesAfterReinit.size(), 2);
    ASSERT_EQ(filenamesAfterReinit[0], new_logs_dir + "/syn_api_log.txt");
    ASSERT_EQ(filenamesAfterReinit[1], new_logs_dir + "/" + newLogFilename);

    // revert back logs directory to one determined based on env vars
    hl_logger::setLogsFolderPathFromEnv();
    // restore original file sink
    hl_logger::setSinks(synapse::LoggerTypes::SYN_API, sinks_orig);
}

TEST(hl_logger_test, check_logs_after_changing_logs_folder)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_TEST, HLLOG_LEVEL_TRACE);
    auto synTestLogFiles = hl_logger::getSinksFilenames(synapse::LoggerTypes::SYN_TEST);

    const std::string synTestLogFilename = filenameFromPath(synTestLogFiles[0]);

    const std::string origLogMsg0 = "check_logs_after_reinit: syn_test 1";
    const std::string origLogMsg1 = "check_logs_after_reinit: syn_test 2";

    const std::string newLogMsg0 = "check_logs_after_reinit: syn_test 3";
    const std::string newLogMsg1 = "check_logs_after_reinit: syn_test 4";
    const std::string newLogMsg2 = "check_logs_after_reinit: syn_test 5";

    HLLOG_DEBUG(SYN_TEST, "{}", origLogMsg0);
    HLLOG_DEBUG(SYN_TEST, "{}", origLogMsg1);
    HLLOG_DEBUG(SYN_TEST, "{}", origLogMsg0);
    hl_logger::flush();

    ASSERT_EQ(findStringsCountInLog(origLogMsg0, synTestLogFilename), 2);
    ASSERT_EQ(findStringsCountInLog(origLogMsg1, synTestLogFilename), 1);

    auto newLogsRoot = hl_logger::getLogsFolderPath() + "/new_logs_root";
    hl_logger::setLogsFolderPath(newLogsRoot);

    ASSERT_EQ(hl_logger::getLogsFolderPath(), newLogsRoot);

    HLLOG_DEBUG(SYN_TEST, "{}", newLogMsg0);
    HLLOG_DEBUG(SYN_TEST, "{}", newLogMsg1);
    HLLOG_DEBUG(SYN_TEST, "{}", newLogMsg2);
    hl_logger::flush();

    ASSERT_EQ(findStringsInLog(origLogMsg0, synTestLogFilename, newLogsRoot).size(), 0);
    ASSERT_EQ(findStringsInLog(origLogMsg1, synTestLogFilename, newLogsRoot).size(), 0);

    ASSERT_EQ(findStringsInLog(newLogMsg0, synTestLogFilename, newLogsRoot).size(), 1);
    ASSERT_EQ(findStringsInLog(newLogMsg1, synTestLogFilename, newLogsRoot).size(), 1);
    ASSERT_EQ(findStringsInLog(newLogMsg2, synTestLogFilename, newLogsRoot).size(), 1);

    fs::remove_all(newLogsRoot);

    // restore logs root dir
    hl_logger::setLogsFolderPathFromEnv();
}

TEST(hl_logger_test, TestLogOffPerf)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_ERROR);
    const int cnt   = 1000000;
    auto      start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cnt; ++i)
    {
        HLLOG_TRACE(SYN_API, "hello {}", 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / double(cnt) << "ns\n";
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);
}

TEST(hl_logger_test, logsFolderISAbsolute)
{
    std::string logsFolder = hl_logger::getLogsFolderPath();
    ASSERT_TRUE(!logsFolder.empty());
    ASSERT_EQ(logsFolder[0], '/');
}

TEST(hl_logger_test, severalLogFiles)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);
    hl_logger::enablePeriodicFlush(true);
    for (unsigned i = 0; i < 200000; ++i)
    {
        HLLOG_TRACE(SYN_API, "log message {} to fill up several log files", i);
    }
}

TEST(hl_logger_test, StackOverflowCrash)
{
    std::string logStr = "before stack overflow " + std::to_string(time(nullptr));
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);

    auto callable = [logStr]() {
        HLLOG_TRACE(SYN_API, "{}", logStr);
        recursion(100);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringsCountInLog(logStr, "syn_api_log.txt")) << "log string not found in the log";
}

// different logger
TEST(hl_logger_test, StackOverflowCrash_SYN_STREAM)
{
    std::string logStr = "before stack overflow SCAL_1 " + std::to_string(time(nullptr));
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_1, HLLOG_LEVEL_TRACE);

    auto callable = [logStr]() {
        HLLOG_TRACE(SCAL_1, "{}", logStr);
        recursion(100);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringsCountInLog(logStr, "log.txt")) << "log string not found in the log";
}

// different file
TEST(hl_logger_test, StackOverflowCrash_GC)
{
    std::string logStr = "before stack overflow GC " + std::to_string(time(nullptr));
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);

    auto callable = [logStr]() {
        HLLOG_TRACE(SYN_API, "{}", logStr);
        recursion(100);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringsCountInLog(logStr, "syn_api_log.txt")) << "log string not found in the log";
}

void writeToPtr(int* ptr)
{
    *ptr = 55;
}

TEST(hl_logger_test, NullPtrCrash)
{
    std::string logStr = "before null ptr " + std::to_string(time(nullptr));
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);
    int* ptr = nullptr;

    auto callable = [logStr, ptr]() {
        HLLOG_TRACE(SYN_API, "{}", logStr);
        writeToPtr(ptr);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringsCountInLog(logStr, "syn_api_log.txt")) << "log string not found in the log";
}

TEST(hl_logger_test, IllegalPtrCrash)
{
    int* ptr = (int*)0xFFFFFFFFFFFFFFFF;
    std::string logStr = "before illegal ptr " + std::to_string((uint64_t)ptr) + " " + std::to_string(time(nullptr));
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SYN_API, HLLOG_LEVEL_TRACE);

    auto callable = [logStr, ptr]() {
        HLLOG_TRACE(SYN_API, "{}", logStr);
        writeToPtr(ptr);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringsCountInLog(logStr, "syn_api_log.txt")) << "log string not found in the log";
}

void throwException()
{
    throw 1;
}

TEST(hl_logger_test, Terminate)
{
    std::string logStr = "before terminate " + std::to_string(time(nullptr));
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_1, HLLOG_LEVEL_TRACE);

    auto callable = [logStr]() {
        HLLOG_TRACE(SCAL_1, "{}", logStr);
        struct S
        {
            ~S() { ::throwException(); }
        };
        S s;
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringsCountInLog(logStr, "log.txt")) << "log string not found in the log";
}


TEST(hl_logger_test, LongTimeAfterLog)
{

    std::string logStr = "before long delay " + std::to_string(time(nullptr));
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_1, HLLOG_LEVEL_TRACE);
    hl_logger::enablePeriodicFlush(false);

    HLLOG_TRACE(SCAL_1, "{}", logStr);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ASSERT_FALSE(findStringsCountInLog(logStr, "log.txt")) << "log string not found in the log";
    hl_logger::enablePeriodicFlush();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ASSERT_TRUE(findStringsCountInLog(logStr, "log.txt")) << "log string not found in the log";

}

TEST(hl_logger_test, ForkFlushing)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_1, HLLOG_LEVEL_TRACE);
    hl_logger::enablePeriodicFlush(false);
    std::string logStr = "before fork " + std::to_string(time(nullptr));
    HLLOG_TRACE(SCAL_1, "{}", logStr);
    int pid = fork();
    hl_logger::flush();
    if (pid == 0)
    {
        exit(0);
    }
    // make sure child process finished - wait a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_EQ(findStringsCountInLog(logStr, "log.txt"), 1) << "only one log string must be in the log";
}

TEST(hl_logger_test, ForkPidTid)
{
    hl_logger::setLoggingLevel(synapse::LoggerTypes::SCAL_API, HLLOG_LEVEL_TRACE);
    hl_logger::enablePeriodicFlush(false);
    std::string logStr = "after fork " + std::to_string(time(nullptr));
    int pid = fork();
    HLLOG_TRACE(SCAL_API, "{}", logStr);
    hl_logger::flush();
    if (pid == 0)
    {
        exit(0);
    }
    // make sure child process finished - wait a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto matchLines = findStringsInLog(logStr, "scal_api_log.txt");
    ASSERT_EQ(matchLines.size(), 2) << "only one log string must be in the log";
    std::vector<std::string> pids;
    for (auto const & line : matchLines)
    {
        std::string pidStr = "[pid:";
        auto pidStart = line.find(pidStr) + pidStr.size();
        ASSERT_NE(pidStart, std::string::npos);
        auto pidEnd = line.find("]", pidStart);
        ASSERT_NE(pidEnd, std::string::npos);
        pids.push_back(line.substr(pidStart , pidEnd - pidStart));
    }
    ASSERT_NE(pids[0], pids[1]);
    std::vector<std::string> tids;
    for (auto const & line : matchLines)
    {
        std::string tidStr = "[tid:";
        auto tidStart = line.find(tidStr) + tidStr.size();
        ASSERT_NE(tidStart, std::string::npos);
        auto tidEnd = line.find("]", tidStart);
        ASSERT_NE(tidEnd, std::string::npos);
        tids.push_back(line.substr(tidStart , tidEnd - tidStart));
    }
    ASSERT_NE(tids[0], tids[1]);
}

static OperationFullMeasurementResults testLongMessageTrace(TestParams params)
{
    unsigned i   = 0;
    static constexpr auto func = HLLOG_FUNC;
    auto     res = measure(
            [&i](unsigned thread_id) {
                HLLOG_TRACE(SYN_API,
                          "{}: Patching (DC {} PP-Index in Stage {:x}) DC-Address 0x{:x} Offset-in-DC 0x{:x},"
                          " section {}, effective-address 0x{:x} patched-location 0x{:x} patch-type {} value 0x{:x}",
                          func,
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
            "long msg buffer trace",
            params);

    return res;
}

static OperationFullMeasurementResults testMediumMessageTrace(TestParams params)
{
    unsigned i   = 0;
    static constexpr auto func = HLLOG_FUNC;
    auto     res = measure(
            [&i](unsigned thread_id) {
                HLLOG_TRACE(SYN_API,
                          "{} section {:x} calc {:x} expected {:x} valid hi/low {:x}/{:x}",
                          func,
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

static OperationFullMeasurementResults testShortMessageTrace(TestParams params)
{
    unsigned i   = 0;
    auto     res = measure(
            [&i](unsigned thread_id) {
                HLLOG_TRACE(SYN_API, "DW HIGH: Current val {:x} different from previous {:x}", i, i + 2);
                ++i;
            },
            "short msg buffer trace",
            params);

    return res;
}

static OperationFullMeasurementResults testShortMessageTraceUntyped(TestParams params)
{
    unsigned i   = 0;
    auto logger = hl_logger::getLogger(synapse::LoggerTypes::SYN_API);
    auto     res = measure(
            [&i, &logger](unsigned thread_id) {
                HLLOG_UNTYPED(logger, HLLOG_LEVEL_TRACE, "DW HIGH: Current val {:x} different from previous {:x}", i, i + 2);
                ++i;
            },
            "short msg buffer trace untyped",
            params);

    return res;
}
static OperationFullMeasurementResults testShortMessageWithString(TestParams params)
{
    unsigned i   = 0;
    auto logger = hl_logger::getLogger(synapse::LoggerTypes::SYN_API);
    auto     res = measure(
            [&i, &logger](unsigned thread_id) {
                HLLOG_UNTYPED(logger, HLLOG_LEVEL_TRACE, "{}: DW HIGH: Current val {:x} different from previous {:x}", "str", i, i + 2);
                ++i;
            },
            "short msg buffer trace with str",
            params);

    return res;
}
TEST(hl_logger_test, DISABLED_logger_performance)
{
    hl_logger::enablePeriodicFlush();
    HLLOG_SET_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_TRACE);
    TestParams params;
    params.maxNbThreads            = 2;
    params.minNbThreads            = 1;
    params.nbTests                 = 100;
    params.internalLoopInterations = 10000;
    params.interations_per_sleep   = 0;
    PrintTestResults("logger_performance",
                     {testLongMessageTrace(params), testMediumMessageTrace(params), testShortMessageTrace(params), testShortMessageTraceUntyped(params)});
    hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LoggerTypes::DFA));
}

TEST(hl_logger_test, logger_performance_lazy)
{
    hl_logger::enablePeriodicFlush();
    HLLOG_SET_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_ERROR);
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_TRACE);
    TestParams params;
    params.maxNbThreads            = 2;
    params.minNbThreads            = 1;
    params.nbTests                 = 100;
    params.internalLoopInterations = 10000;
    params.interations_per_sleep   = 0;

    PrintTestResults("lazy logger_performance",
                     {testLongMessageTrace(params), testMediumMessageTrace(params), testShortMessageTrace(params), testShortMessageTraceUntyped(params), testShortMessageWithString(params)});
    hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LoggerTypes::DFA));
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_OFF);
}

TEST(hl_logger_test, lazy_log_basic)
{
    hl_logger::enablePeriodicFlush();
    HLLOG_SET_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_OFF);
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_TRACE);
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API2, HLLOG_LEVEL_TRACE);
    //HLLOG_SET_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_TRACE);

    HLLOG_DEBUG(SYN_API, "basic lazy log line {}", std::string("hello world a loooong greeeting ").c_str());
    HLLOG_DEBUG(SYN_API, "basic lazy log line {}", "hello world a loooong greeeting 2");
    HLLOG_DEBUG(SYN_API2, "api2 basic lazy log line {}", "hello world a loooong greeeting 2");
    HLLOG_DEBUG(SYN_API, "basic lazy log line {}", "hello world a loooong greeeting 3");
    HLLOG_DEBUG(SYN_API, "basic lazy log line {}", HclImmediateData(5));
    HLLOG_DEBUG(SYN_API, "basic lazy log line");
    HLLOG_DEBUG(SYN_API, "basic lazy log line");
    HLLOG_DEBUG(SYN_API, "basic lazy log line");
    HLLOG_DEBUG(SYN_API, "basic lazy log line");
    HLLOG_DEBUG(SYN_API, "basic lazy log line");
    HLLOG_DEBUG_F(SYN_API, "basic lazy log line");
    auto apiLogger = hl_logger::getLogger(synapse::LoggerTypes::SYN_API);
    for (unsigned i = 0; i < 10; ++i)
    {
        HLLOG_UNTYPED(apiLogger, HLLOG_LEVEL_DEBUG, "untyped basic lazy log line {} {}", i, i + 1);
    }


    hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LoggerTypes::DFA));
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_OFF);
}

TEST(hl_logger_test, large_lazy_msg)
{
    hl_logger::enablePeriodicFlush();
    HLLOG_SET_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_ERROR);
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_TRACE);

    for (unsigned i = 0 ; i < 4000; ++i)
    {
        HLLOG_DEBUG(SYN_API, "large lazy log line {} {} {} {} {}", "1", "2", "3", "4", "5");
        HLLOG_DEBUG(SYN_API, "short lazy log line {}", 1);
    }

    hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LoggerTypes::DFA));
    HLLOG_SET_LAZY_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_OFF);
}

TEST(hl_logger_test, logger_performance_no_logs)
{
    HLLOG_SET_LOGGING_LEVEL(SYN_API, HLLOG_LEVEL_ERROR);
    TestParams params;
    params.maxNbThreads            = 1;
    params.minNbThreads            = 1;
    params.nbTests                 = 100;
    params.internalLoopInterations = 10000;
    params.interations_per_sleep   = 0;
    PrintTestResults("logger_performance",
                     {testLongMessageTrace(params), testMediumMessageTrace(params), testShortMessageTrace(params)});
}