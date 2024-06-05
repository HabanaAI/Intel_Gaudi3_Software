#include "event_triggered_logger.hpp"
#include "synapse_api.h"

#include <gtest/gtest.h>
#include <thread>

bool g_traceModeLogging = false;
#define ENFORCE_TRACE_MODE_LOGGING
#include "log_manager.h"
#include "syn_logging.h"

void testLogLines(uint32_t numOfParams, uint32_t iterations, bool includeString)
{
    const uint32_t MAX_LOG_PARAMS = 5;
    if (numOfParams > MAX_LOG_PARAMS)
    {
        LOG_DEBUG(SYN_RT_TEST, "Test log-params amount limit is {}", MAX_LOG_PARAMS);
        return;
    }

    timespec etlTimerStart, etlTimerEnd;
    timespec spdlTimerStart, spdlTimerEnd;
    double   etlTotal, spdlTotal;

    TURN_ON_TRACE_MODE_LOGGING();

    // ---- ETL (Start)
    spEventTriggeredLoggerManager& eventTriggerLogger = EventTriggeredLoggerManager::getInstance();

    clock_gettime(CLOCK_MONOTONIC_RAW, &etlTimerStart);
    for (uint32_t i = 0; i < iterations; i++)
    {
        uint64_t logId = eventTriggerLogger->preOperation(EVENT_LOGGER_LOG_TYPE_FOR_TESTING, __FUNCTION__);

        // Sadly VA_ARGS is not supported, so we will go the ugly way
        if (includeString)
        {
            switch (numOfParams)
            {
                case 0:
                    // Redundant
                    break;

                case 1:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "One Arg log {}",
                                                    "string");
                    break;

                case 2:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Two Args log {} {}",
                                                    "string",
                                                    1);
                    break;

                case 3:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Three Args log {} {} {}",
                                                    "string",
                                                    1,
                                                    2);
                    break;

                case 4:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Four Args log {} {} {} {}",
                                                    "string",
                                                    1,
                                                    2,
                                                    3);
                    break;

                case 5:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Five Args log {} {} {} {} {}",
                                                    "string",
                                                    1,
                                                    2,
                                                    3,
                                                    4);
                    break;
            }
        }
        else
        {
            switch (numOfParams)
            {
                case 0:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING, logId, "No Args log");
                    break;

                case 1:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING, logId, "One Arg log {}", 0);
                    break;

                case 2:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Two Args log {} {}",
                                                    0,
                                                    1);
                    break;

                case 3:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Three Args log {} {} {}",
                                                    0,
                                                    1,
                                                    2);
                    break;

                case 4:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Four Args log {} {} {} {}",
                                                    0,
                                                    1,
                                                    2,
                                                    3);
                    break;

                case 5:
                    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                    logId,
                                                    "Five Args log {} {} {} {} {}",
                                                    0,
                                                    1,
                                                    2,
                                                    3,
                                                    4);
                    break;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &etlTimerEnd);
    etlTotal = (etlTimerEnd.tv_nsec - etlTimerStart.tv_nsec) / 1000.0 +
               (etlTimerEnd.tv_sec - etlTimerStart.tv_sec) * 1000000.0;
    // ---- ETL (End)

    // ---- SPDL (Start)
    clock_gettime(CLOCK_MONOTONIC_RAW, &spdlTimerStart);
    for (uint32_t i = 0; i < iterations; i++)
    {
        // Sadly VA_ARGS is not supported, so we will go the ugly way

        if (includeString)
        {
            switch (numOfParams)
            {
                case 0:
                    LOG_TRACE(SYN_RT_TEST, "No Args log");
                    break;
                case 1:
                    LOG_TRACE(SYN_RT_TEST, "One Arg log {}", "string");
                    break;
                case 2:
                    LOG_TRACE(SYN_RT_TEST, "Two Args log {} {}", "string", 1);
                    break;
                case 3:
                    LOG_TRACE(SYN_RT_TEST, "Three Args log {} {} {}", "string", 1, 2);
                    break;
                case 4:
                    LOG_TRACE(SYN_RT_TEST, "Four Args log {} {} {} {}", "string", 1, 2, 3);
                    break;
                case 5:
                    LOG_TRACE(SYN_RT_TEST, "Five Args log {} {} {} {} {}", "string", 1, 2, 3, 4);
                    break;
            }
        }
        else
        {
            switch (numOfParams)
            {
                case 0:
                    LOG_TRACE(SYN_RT_TEST, "No Args log");
                    break;
                case 1:
                    LOG_TRACE(SYN_RT_TEST, "One Arg log {}", 0);
                    break;
                case 2:
                    LOG_TRACE(SYN_RT_TEST, "Two Args log {} {}", 0, 1);
                    break;
                case 3:
                    LOG_TRACE(SYN_RT_TEST, "Three Args log {} {} {}", 0, 1, 2);
                    break;
                case 4:
                    LOG_TRACE(SYN_RT_TEST, "Four Args log {} {} {} {}", 0, 1, 2, 3);
                    break;
                case 5:
                    LOG_TRACE(SYN_RT_TEST, "Five Args log {} {} {} {} {}", 0, 1, 2, 3, 4);
                    break;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &spdlTimerEnd);
    spdlTotal = (spdlTimerEnd.tv_nsec - spdlTimerStart.tv_nsec) / 1000.0 +
                (spdlTimerEnd.tv_sec - spdlTimerStart.tv_sec) * 1000000.0;
    // ---- SPDL (End)

    TURN_OFF_TRACE_MODE_LOGGING();

    LOG_DEBUG(SYN_RT_TEST,
              "\t\t Num-Of-Params {}: ETL {:{}.{}f}us ({:{}.{}f}us) SPDL {:{}.{}f}us ({:{}.{}f}us)",
              numOfParams,
              etlTotal,
              3, /* format-width     */
              3, /* format-precision */
              etlTotal / iterations,
              2, /* format-width     */
              5, /* format-precision */
              spdlTotal,
              3, /* format-width     */
              3, /* format-precision */
              spdlTotal / iterations,
              2, /* format-width     */
              5 /* format-precision */);
}

// This test is defined to ensure that the ETL will not be broken
TEST(UTEtlTest, event_trigger_logger_usage)
{
    bool status = false;

    TURN_ON_TRACE_MODE_LOGGING();

    const uint32_t eventLoggerSize = 100;

    class TestExecutor : public EventTriggeredExecutor
    {
        virtual void triggerEventExecution(eEventLoggerTriggerType logTriggerType) override {
            // Do nothing
        };

        virtual std::string getName() override { return std::string("Test executor"); };
    };
    TestExecutor testExecutor;

    spEventTriggeredLoggerManager& eventTriggerLogger = EventTriggeredLoggerManager::getInstance();

    // Ensure there will be no loggers nor executors exists, by the ETL, prior of the testing
    eventTriggerLogger->clear();

    status = eventTriggerLogger->createLogger(EVENT_LOGGER_LOG_TYPE_CS_ORDER, EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER, 10);
    ASSERT_EQ(true, status) << "Failed to create logger (1)";

    eventTriggerLogger->ignoreLogger(EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    status = eventTriggerLogger->createLogger(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                              EVENT_LOGGER_TRIGGER_TYPE_FOR_TESTING,
                                              eventLoggerSize);
    ASSERT_EQ(true, status) << "Failed to create logger (2)";

    status = eventTriggerLogger->releaseLogger(EVENT_LOGGER_LOG_TYPE_CS_ORDER);
    ASSERT_EQ(true, status) << "Failed to release logger (1)";

    status = eventTriggerLogger->addExecutor(&testExecutor);
    ASSERT_EQ(true, status) << "Failed to add executor";

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_FOR_TESTING);
    eventTriggerLogger->addLoggingV(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                    logId,
                                    "Test {} with value 0x{:x}",
                                    "\"test's-text\"",
                                    1);

    ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_FOR_TESTING);

    status = eventTriggerLogger->removeExecutor(&testExecutor);
    ASSERT_EQ(true, status) << "Failed to remove executor";

    status = eventTriggerLogger->releaseLogger(EVENT_LOGGER_LOG_TYPE_FOR_TESTING);
    ASSERT_EQ(true, status) << "Failed to release logger (2)";

    TURN_OFF_TRACE_MODE_LOGGING();

    EventTriggeredLoggerManager::getInstance()->releaseInstance();
}

// Compares ETL log-line cost compared to regular log-line
TEST(UTEtlTest, event_trigger_logger_cost)
{
    CREATE_LOGGER(SYN_RT_TEST, SYNAPSE_LOG_SINK_FILE, 1024 * 1024, 5);

    // Ensure there will be no loggers nor executors exists, by the ETL, prior of the testing
    static const uint32_t          etlLoggerSize      = 500;
    spEventTriggeredLoggerManager& eventTriggerLogger = EventTriggeredLoggerManager::getInstance();

    eventTriggerLogger->clear();
    bool status = eventTriggerLogger->createLogger(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                   EVENT_LOGGER_TRIGGER_TYPE_FOR_TESTING,
                                                   etlLoggerSize);
    ASSERT_EQ(true, status) << "Failed to create logger";

    const uint32_t iterations = 10000;
    LOG_DEBUG(SYN_RT_TEST, "Average over {} iterations;", iterations);

    uint32_t numOfParams   = 1;
    bool     includeString = true;
    LOG_DEBUG(SYN_RT_TEST, "\t With string parameter:");
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);

    numOfParams   = 0;
    includeString = false;
    LOG_DEBUG(SYN_RT_TEST, "\t Only numeric parameters:");
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);
    testLogLines(numOfParams++, iterations, includeString);

    status = eventTriggerLogger->releaseLogger(EVENT_LOGGER_LOG_TYPE_FOR_TESTING);
    ASSERT_EQ(true, status) << "Failed to release logger";

    EventTriggeredLoggerManager::getInstance()->releaseInstance();

    DROP_LOGGER(SYN_RT_TEST);
}

TEST(UTEtlTest, check_interleaved_logs_and_triggers)
{
    CREATE_LOGGER(SYN_RT_TEST, SYNAPSE_LOG_SINK_FILE, 1024 * 1024, 5);

    TURN_ON_TRACE_MODE_LOGGING();

    // Ensure there will be no loggers nor executors exists, by the ETL, prior of the testing
    static const uint32_t          etlLoggerSize      = 500;
    spEventTriggeredLoggerManager& eventTriggerLogger = EventTriggeredLoggerManager::getInstance();
    eventTriggerLogger->clear();

    bool status = eventTriggerLogger->createLogger(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                                                   EVENT_LOGGER_TRIGGER_TYPE_FOR_TESTING,
                                                   etlLoggerSize);

    auto loggingWork = [&](uint32_t threadIndex) {
        static const uint32_t amountOfLogLines = 10000;

        uint32_t logCounter = 0;
        while (logCounter != amountOfLogLines)
        {
            usleep(10);
            ETL_TRACE(EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
                      SYN_RT_TEST,
                      "Thread Index: {}, Log line: {}",
                      threadIndex,
                      logCounter);
            logCounter++;
        }
    };

    auto triggerWork = [&]() {
        static const uint32_t amountOfTriggers = 120;

        uint32_t triggerCounter = 0;
        while (triggerCounter != amountOfTriggers)
        {
            status = ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_FOR_TESTING);
            ASSERT_EQ(status, true) << "Trigger failure";
            usleep(50);
            triggerCounter++;
        };
    };

    const uint32_t               numOfLoggingThreads = 10;
    std::vector<std::thread>     logThread;
    std::shared_ptr<std::thread> pTriggerThread;

    logThread.reserve(numOfLoggingThreads);
    for (uint32_t i = 0; i < numOfLoggingThreads; i++)
    {
        std::thread currentThread(loggingWork, i);
        logThread.push_back(std::move(currentThread));
    }

    for (uint32_t i = 0; i < numOfLoggingThreads; i++)
    {
        logThread[i].join();
    }

    pTriggerThread = std::make_shared<std::thread>(triggerWork);
    if (pTriggerThread != nullptr)  // For safe-keeping...
    {
        pTriggerThread->join();
    }

    status = eventTriggerLogger->releaseLogger(EVENT_LOGGER_LOG_TYPE_FOR_TESTING);
    ASSERT_EQ(true, status) << "Failed to release logger";

    TURN_OFF_TRACE_MODE_LOGGING();

    EventTriggeredLoggerManager::getInstance()->releaseInstance();

    DROP_LOGGER(SYN_RT_TEST);
}