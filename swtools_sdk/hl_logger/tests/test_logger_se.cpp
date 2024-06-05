#include <gtest/gtest.h>
#include "hl_logger/hllog_se.hpp"
#include <fstream>
#include "utils.h"

TEST(hl_logger_test_se, basic)
{
    CREATE_LOGGER("LOGGER_SE", "logger_se.log", 1000 * 1024 * 5, 1);
    SET_LOG_LEVEL("LOGGER_SE", 0);
    LOG_TRACE("LOGGER_SE", "hello world from old {}", "api");
    LOG_TRACE("LOGGER_SE_WRONG_NAME", "hello world from old {}", "api");
    DROP_ALL_LOGGERS;
}

TEST(hl_logger_test_se, crash_test)
{
    std::string logStr = "before stack overflow LOGGER_SE " + std::to_string(time(nullptr));
    CREATE_LOGGER("LOGGER_SE", "logger_se.log", 1000 * 1024 * 5, 1);
    SET_LOG_LEVEL("LOGGER_SE", 0);
    auto callable = [logStr]() {
        LOG_TRACE("LOGGER_SE", "{}", logStr);
        recursion(100);
    };

    ASSERT_DEATH(callable(), "");
    ASSERT_TRUE(findStringsCountInLog(logStr, "logger_se.log")) << "log string not found in the log";
    DROP_ALL_LOGGERS;
}

TEST(hl_logger_test_se, wrong_format_test)
{
    CREATE_LOGGER("LOGGER_SE", "logger_se.log", 1000 * 1024 * 5, 1);
    SET_LOG_LEVEL("LOGGER_SE", 0);
    LOG_TRACE("LOGGER_SE", "{} {}", 1);
}