#include "hl_logger/hllog.hpp"

// copy an old version LoggerCreateParams member to a new version
#define ASSIGN(fieldName) pV_new.fieldName = static_cast<decltype(pV_new.fieldName)>(params.fieldName)

namespace hl_logger{

inline namespace v1_2{
struct LoggerCreateParams
{
    std::string logFileName;                      // main log file. rotates and preserves previous log messages
    unsigned    logFileSize         = 0;          // max log file
    unsigned    logFileAmount       = 1;          // number of files for rotation
    bool        rotateLogfileOnOpen = false;      // rotate logFile on logger creation
    uint64_t    logFileBufferSize   = 0;          // default value (~5MB). if LOG_FILE_SIZE envvar is set - use its value
    std::string separateLogFile;                  // a separate log file (if needed). it's recreated on each createLogger call
    uint64_t    separateLogFileBufferSize = 0;
    bool        registerLogger      = false;      // register logger in the global registry (enable access by name from different modules)
    bool        sepLogPerThread     = false;      // separate log file per thread
    bool        printSpecialContext = false;      // print special context [C:] for each log message
    bool        printThreadID       = true;       // print tid [tid:<TID>] for each log message
    bool        printProcessID      = false;      // print pid [pid:<PID>] for each log message
    bool        forcePrintFileLine  = false;      // if false - print if PRINT_FILE_AND_LINE envvar is true
    bool        printTime           = true;
    bool        printLoggerName     = true;

    enum class LogLevelStyle
    {
        off,
        full_name, // [trace][debug][info][warning][error][critical]
        one_letter // [T][D][I][W][E][C]
    };
    LogLevelStyle logLevelStyle    = LogLevelStyle::full_name;
    std::string   spdlogPattern;                       // default(empty): [time][loggerName][Level] msg
    unsigned      loggerNameLength = 0;                // default(0): max length of all the logger names
    int           loggerFlushLevel = HLLOG_LEVEL_WARN; // only messages with at least loggerFlushLevel are flushed immediately
    // only messages with at least loggingLevel are printed
    // logLevel is :
    // 1. LOG_LEVEL_<LOGGER_NAME> envvar (if it's set). if it's not set see 2.
    // 2. LOG_LEVEL_ALL_<LOGGER_PREFIX> envvar (if it's set). if it's not set - defaultLogLevel
    int         defaultLoggingLevel          = HLLOG_LEVEL_CRITICAL;
    bool        forceDefaultLoggingLevel     = false;    // ignore envvars and set logLevel to defaultLogLevel
    int         defaultLazyLoggingLevel      = HLLOG_LEVEL_OFF;
    bool        forceDefaultLazyLoggingLevel = false;    // ignore envvars and set logLevel to defaultLogLevel
    uint32_t    defaultLazyQueueSize         = HLLOG_DEFAULT_LAZY_QUEUE_SIZE; // default size of lazy log messages queue
    enum class ConsoleStream
    {
        std_out,
        std_err,
        disabled
    };
    ConsoleStream consoleStream = ConsoleStream::std_out;  // type of console stream if ENABLE_CONSOLE envvar is on
};

HLLOG_API LoggerSPtr createLogger(std::string_view loggerName, LoggerCreateParams const& params)
{
    hl_logger::v1_3::LoggerCreateParams pV_new;
    ASSIGN(logFileName);
    ASSIGN(logFileSize);
    ASSIGN(logFileAmount);
    ASSIGN(rotateLogfileOnOpen);
    ASSIGN(logFileBufferSize);
    ASSIGN(separateLogFile);
    ASSIGN(separateLogFileBufferSize);
    ASSIGN(registerLogger);
    ASSIGN(sepLogPerThread);
    ASSIGN(printSpecialContext);
    ASSIGN(printThreadID);
    ASSIGN(printProcessID);
    ASSIGN(forcePrintFileLine);
    ASSIGN(spdlogPattern);
    ASSIGN(loggerNameLength);
    ASSIGN(loggerFlushLevel);
    ASSIGN(defaultLoggingLevel);
    ASSIGN(forceDefaultLoggingLevel);
    ASSIGN(defaultLazyLoggingLevel);
    ASSIGN(forceDefaultLazyLoggingLevel);
    ASSIGN(defaultLazyQueueSize);
    ASSIGN(printTime);
    ASSIGN(printLoggerName);
    ASSIGN(logLevelStyle);
    ASSIGN(consoleStream);
    return hl_logger::v1_3::createLogger(loggerName, pV_new);
}
}

// binary compatibility with v1.1
inline namespace v1_1{
struct LoggerCreateParams
{
    std::string logFileName;                      // main log file. rotates and preserves previous log messages
    unsigned    logFileSize         = 0;          // max log file
    unsigned    logFileAmount       = 1;
    uint64_t    logFileBufferSize   = 0;           // default value (~5MB). if LOG_FILE_SIZE envvar is set - use its value
    std::string separateLogFile;                   // a separate log file (if needed). it's recreated on each createLogger call
    uint64_t    separateLogFileBufferSize = 0;
    bool        registerLogger      = false;       // register logger in the global registry
    bool        sepLogPerThread     = false;       // separate log file per thread
    bool        printSpecialContext = false;       // print special context [C:] for each log message
    bool        printThreadID       = true;        // print tid [tid:<TID>] for each log message
    bool        printProcessID      = false;       // print pid [pid:<PID>] for each log message
    bool        forcePrintFileLine  = false;       // if false - print if PRINT_FILE_AND_LINE envvar is true
    bool        printTime           = true;
    bool        printLoggerName     = true;
    enum class LogLevelStyle
    {
        off,
        full_name, // [trace][debug][info][warning][error][critical]
        one_letter // [T][D][I][W][E][C]
    };
    LogLevelStyle logLevelStyle    = LogLevelStyle::full_name;
    std::string   spdlogPattern;                       // default(empty): [time][loggerName][Level] msg
    unsigned      loggerNameLength = 0;                // default(0): max length of all the logger names
    int           loggerFlushLevel = HLLOG_LEVEL_WARN; // only messages with at least loggerFlushLevel are flushed immediately
    // only messages with at least loggingLevel are printed
    // logLevel is :
    // 1. LOG_LEVEL_<LOGGER_NAME> envvar (if it's set). if it's not set see 2.
    // 2. LOG_LEVEL_ALL_<LOGGER_PREFIX> envvar (if it's set). if it's not set - defaultLogLevel
    int         defaultLoggingLevel      = HLLOG_LEVEL_CRITICAL;
    bool        forceDefaultLoggingLevel = false;  // ignore envvars and set logLevel to defaultLogLevel
    enum class ConsoleStream
    {
        std_out,
        std_err,
        disabled
    };
    ConsoleStream consoleStream = ConsoleStream::std_out;  // type of console stream if ENABLE_CONSOLE envvar is on
};

HLLOG_API LoggerSPtr createLogger(std::string_view loggerName, LoggerCreateParams const& params)
{
    hl_logger::v1_2::LoggerCreateParams pV_new;
    ASSIGN(logFileName);
    ASSIGN(logFileSize);
    ASSIGN(logFileAmount);
    ASSIGN(logFileBufferSize);
    ASSIGN(separateLogFile);
    ASSIGN(separateLogFileBufferSize);
    ASSIGN(registerLogger);
    ASSIGN(sepLogPerThread);
    ASSIGN(printSpecialContext);
    ASSIGN(printThreadID);
    ASSIGN(printProcessID);
    ASSIGN(forcePrintFileLine);
    ASSIGN(spdlogPattern);
    ASSIGN(loggerNameLength);
    ASSIGN(loggerFlushLevel);
    ASSIGN(defaultLoggingLevel);
    ASSIGN(forceDefaultLoggingLevel);
    ASSIGN(printTime);
    ASSIGN(printLoggerName);
    ASSIGN(logLevelStyle);
    ASSIGN(consoleStream);
    return hl_logger::v1_2::createLogger(loggerName, pV_new);
}
}

// binary compatibility with v1.0
inline namespace v1_0 {
struct LoggerCreateParams {
    // main log file
    std::string logFileName;
    unsigned    logFileSize = 0;
    unsigned    logFileAmount = 1;
    uint64_t    logFileBufferSize = 0;           // default value (~5MB). if LOG_FILE_SIZE envvar is set - use its value
    std::string separateLogFile;                   // a separate log file (if needed)
    uint64_t    separateLogFileBufferSize = 0;
    bool        registerLogger = false;       // register logger in the global registry
    bool        sepLogPerThread = false;       // separate log file per thread
    bool        printSpecialContext = false;       // print special context [C:] for each log message
    bool        printThreadID = true;        // print tid [tid:<TID>] for each log message
    bool        printProcessID = false;       // print pid [pid:<PID>] for each log message
    bool        forcePrintFileLine = false;       // if false - print if PRINT_FILE_AND_LINE envvar is true
    std::string spdlogPattern;                     // default(empty): [time][loggerName][Level] msg
    unsigned    loggerNameLength = 0;              // default(0): max length of all the logger names
    int         loggerFlushLevel = HLLOG_LEVEL_WARN; // only messages with at least loggerFlushLevel are flushed immediately
    // only messages with at least loggingLevel are printed
    // logLevel is :
    // 1. LOG_LEVEL_<LOGGER_NAME> envvar (if it's set). if it's not set see 2.
    // 2. LOG_LEVEL_ALL_<LOGGER_PREFIX> envvar (if it's set). if it's not set - defaultLogLevel
    int         defaultLoggingLevel = HLLOG_LEVEL_CRITICAL;
    bool        forceDefaultLoggingLevel = false;  // ignore envvars and set logLevel to defaultLogLevel
    enum class ConsoleStream {
        std_out,
        std_err,
        disabled
    };
    ConsoleStream consoleStream = ConsoleStream::std_out;  // type of console stream if ENABLE_CONSOLE envvar is on
};

[[maybe_unused]] HLLOG_API LoggerSPtr createLogger(std::string_view loggerName, LoggerCreateParams const &params)
{
    hl_logger::v1_1::LoggerCreateParams pV_new;
    ASSIGN(logFileName);
    ASSIGN(logFileSize);
    ASSIGN(logFileAmount);
    ASSIGN(logFileBufferSize);
    ASSIGN(separateLogFile);
    ASSIGN(separateLogFileBufferSize);
    ASSIGN(registerLogger);
    ASSIGN(sepLogPerThread);
    ASSIGN(printSpecialContext);
    ASSIGN(printThreadID);
    ASSIGN(printProcessID);
    ASSIGN(forcePrintFileLine);
    ASSIGN(spdlogPattern);
    ASSIGN(loggerNameLength);
    ASSIGN(loggerFlushLevel);
    ASSIGN(defaultLoggingLevel);
    ASSIGN(forceDefaultLoggingLevel);
    ASSIGN(consoleStream);
    return hl_logger::v1_1::createLogger(loggerName, pV_new);
}
}
}