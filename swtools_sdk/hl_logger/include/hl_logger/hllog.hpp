#pragma once
#include <array>
#include <memory>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <magic_enum-0.8.1/include/magic_enum.hpp>
#include "hllog_core.hpp"
#include <unordered_map>

#if defined(FMT_VERSION) && FMT_VERSION != 90100
#error "fmt of an incompatible version was already included. only one version of fmt is allowed in one translation unit"
#endif

#include "impl/hllog_fmt_headers.hpp"

#define HLLOG_UNLIKELY(x) __builtin_expect((x), 0)
#define HLLOG_LIKELY(x) __builtin_expect((x), 1)

// use HLLOG_FUNC instead of __FUNCTION__ to get good performance for lazy logs
#define HLLOG_FUNC  hl_logger::static_string(__func__)
HLLOG_BEGIN_NAMESPACE

/**
 * @brief createLogger create a logger
 * @param loggerEnumItem logger enum item
 * @param params logger creation parameters
 */
template<class TLoggerEnum>
inline void createLogger(TLoggerEnum loggerEnumItem, hl_logger::LoggerCreateParams const& params);

/**
 * @brief createLoggerOnDemand create an on demand logger.
 *        Its log file is created when the first message is logged into this logger
 * @param loggerEnumItem logger enum item
 * @param params logger creation parameters
 */
template<class TLoggerEnum>
inline void createLoggerOnDemand(TLoggerEnum loggerEnumItem, hl_logger::LoggerCreateParams const& params);

/**
 * @brief createLoggersOnDemand create on demand loggers that have the same parameters.
 *        Their log file is created when the first message is logged into this logger
 * @param loggerEnumItems logger enum items
 * @param params logger creation parameters
 */
template<class TLoggerEnum>
inline void createLoggersOnDemand(std::initializer_list<TLoggerEnum> const& loggerEnumItems,
                                  hl_logger::LoggerCreateParams const&      params);

/**
 * @brief createLoggers create loggers that log into the same log file
 * @param loggerEnumItems logger enum items
 * @param params loggers creation parameters
 */
template<class TLoggerEnum>
inline void createLoggers(std::initializer_list<TLoggerEnum> const& loggerEnumItems,
                          hl_logger::LoggerCreateParams const&      params);

/**
 * @brief setLoggingLevel set minimal log level. all the messages with log level lower that newLevel are ignored
 * @param loggerEnumItem logger enum item
 * @param newLevel       new log level (HLLOG_LEVEL_TRACE ... HLLOG_LEVEL_OFF)
 */
template<class TLoggerEnum>
void setLoggingLevel(TLoggerEnum loggerEnumItem, int newLevel);

/**
 * @brief get logging level of the logger loggerEnumItem
 * @param loggerEnumItem logger enum item
 * @return current logging level
 */
template<class TLoggerEnum>
int getLoggingLevel(TLoggerEnum loggerEnumItem);

/**
 * @brief check if log level of a logger is not more than level. so that a message with level will be logged
 * @param loggerEnumItem logger enum item
 * @param level          level to compare with
 * @return true if logger's logLevel <= level
 */
template<class TLoggerEnum>
inline bool logLevelAtLeast(TLoggerEnum loggerEnumItem, int level);

/**
 * @brief getLogger get a logger shared_ptr that can be used with hllog_core.hpp api
 * @param loggerEnumItem logger enum item
 * @return logger
 */
template<class TLoggerEnum>
inline hl_logger::LoggerSPtr getLogger(TLoggerEnum loggerEnumItem);

/**
 * @brief isLoggerInstantiated checks if onDemand logger was instantiated
 * @param loggerEnumItem logger enum item
 * @return true if onDemand logger was instantiated
 */
template<class TLoggerEnum>
inline bool isLoggerInstantiated(TLoggerEnum loggerEnumItem);

/**
 * @brief format and log message into logger
 * @param logger
 * @param logLevel           log level
 * @param forcePrintFileLine print source filename and line
 * @param file               source code filename
 * @param line               source code line
 * @param fmtMsg             format message
 * @param args               arguments for formatting
 */
template<typename TFmtMsg, typename... Args>
inline void log(LoggerSPtr const & logger, int logLevel, bool forcePrintFileLine, std::string_view file, int line, TFmtMsg fmtMsg, Args&&... args);

/**
 * @brief logStacktrace log stack trace
 * @param loggerEnumItem logger enum item
 * @param level log level
 */
template<class TLoggerEnum>
inline void logStacktrace(const TLoggerEnum loggerEnumItem, int level);

/**
 * @brief getSinks get logger sinks
 *        NOT THREAD-SAFE
 * @param loggerEnumItem logger enum item
 * @return sinks of the logger
 */
template<class TLoggerEnum>
inline SinksSPtr getSinks(const TLoggerEnum loggerEnumItem);

/**
 * @brief getSinksFilenames
 *        NOT THREAD SAFE
 * @param loggerEnumItem logger enum item
 * @return filenames of file_sinks that are connected to the logger
 */
template<class TLoggerEnum>
HLLOG_API std::vector<std::string> getSinksFilenames(const TLoggerEnum loggerEnumItem);

/**
 * @brief setSinks set logger sinks
 *        NOT THREAD-SAFE
 * @param loggerEnumItem logger enum item
 * @param sinks          new sinks for a logger
 * @return previous sinks of the logger
 */
template<class TLoggerEnum>
inline SinksSPtr setSinks(const TLoggerEnum loggerEnumItem, SinksSPtr sinks = SinksSPtr());

/**
 * @brief addFileSink add a file sink to an existing logger
 *        NOT THREAD SAFE
 * @param loggerEnumItem logger enum item
 * @param logFileName
 * @param logFileSize
 * @param logFileAmount
 * @param loggingLevel  logging level of the new file sink, by default it's equal to the logging level of the logger
  */
template<class TLoggerEnum>
HLLOG_API void addFileSink(const TLoggerEnum loggerEnumItem,
                           std::string_view  logFileName,
                           size_t            logFileSize,
                           size_t            logFileAmount,
                           int               loggingLevel = defaultLoggingLevel);

/**
 * @brief addConsole add a console sinks to a logger
 *        NOT THREAD-SAFE
 * @param loggerEnumItem logger enum item
 * @return ResourceGuard that will remove added console in its dtor
 */
template<class TLoggerEnum>
inline ResourceGuard addConsole(const TLoggerEnum loggerEnumItem);

/**
 * @brief removeConsole remove a console sinks from a logger
 *        NOT THREAD-SAFE
 * @param loggerEnumItem logger enum item
 * @return true if successfully removed a console sink
 */
template<class TLoggerEnum>
inline bool removeConsole(const TLoggerEnum loggerEnumItem);

/**
 * @brief flush a specific logger
 * @param loggerEnumItem logger enum item
 * @usage flush(LoggerTypesEnum::LOGGER_NAME);
 */
template<class TLoggerEnum>
inline void flush(const TLoggerEnum loggerEnumItem);

/**
 * @brief flushAll flush all loggers of a module
 * @tparam TLoggerEnum logger enum
 * @usage  flushAll<LoggerTypesEnum>();
 */
template<class TLoggerEnum>
inline void flushAll();

template<class TLoggerEnum>
constexpr unsigned getNbLoggers()
{
    return (unsigned)TLoggerEnum::LOG_MAX;
}

/**
 * @brief getLoggerEnumItemName get a string representation of a logger enum item
 *        it's defined inside HLLOG_DEFINE_MODULE_LOGGER
 * @param loggerEnumItem
 * @return string representation of a logger enum item
 */
template<class TLoggerEnum>
std::string_view getLoggerEnumItemName(TLoggerEnum loggerEnumItem);

/**
 * drop a specific logger
 * @param loggerEnumItem
 * @return true if the logger was dropped successfully
 *         if the logger was already dropped - return false
 */
template<class TLoggerEnum>
inline bool drop(const TLoggerEnum loggerEnumItem);

struct ScopedLogContext
{
    ScopedLogContext(const std::string& context) { hl_logger::addCurThreadGlobalContext(context); }
    ~ScopedLogContext() { hl_logger::removeCurThreadGlobalContext(); }

    ScopedLogContext()                        = delete;
    ScopedLogContext(ScopedLogContext const&) = delete;
    ScopedLogContext(ScopedLogContext&&)      = delete;
};
HLLOG_END_NAMESPACE

#include "impl/hllog_macros.hpp"
#include "impl/hllog_fmt_compat_layer.inl"
#include "impl/hllog.inl"

#ifdef WIN32
#define HLLOG_FILENAME (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define HLLOG_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#define HLLOG_DUPLICATE_PARAM hl_logger::details::duplicate

#define HLLOG_DUPLICATE(p) p

#ifdef HLLOG_ENABLE_LAZY_LOGGING
    #define HLLOG_TYPED_LAZY(loggerEnumItem, logLevel, fmtMsg, ...)                                                    \
        if (hl_logger::lazyLogLevelAtLeast(loggerEnumItem, logLevel))                                                  \
        {                                                                                                              \
            hl_logger::logLazy(loggerEnumItem,                                                                         \
                               logLevel,                                                                               \
                               fmtMsg HLLOG_APPLY_WITH_LEADING_COMMA(HLLOG_DUPLICATE_PARAM, ##__VA_ARGS__));           \
        }
#else
    #define HLLOG_TYPED_LAZY(loggerEnumItem, logLevel, fmtMsg, ...)
#endif
/**
 * @brief log into a logger (loggerEnumItem) with all the parameters. with or without fmtMsg compilation
 * @param loggerEnumItem     logger enum item
 * @param logLevel           message log level
 * @param forcePrintFileLine print source code file and line
 * @param filename           source code filename
 * @param line               source code line
 * @param fmtMsg             fmt-lib format message. must be FMT_COMPILE(fmtMsg) for compile. or fmtMsg if no compilation needed
 */
#define HLLOG_TYPED_FULL_PREFIXED(loggerEnumItem, logLevel, forcePrintFileLine, filename, line, fmtMsg, ...)           \
    do{                                                                                                                \
        if (HLLOG_UNLIKELY(hl_logger::anyLogLevelAtLeast(loggerEnumItem, logLevel)))                                   \
        {                                                                                                              \
            if (hl_logger::logLevelAtLeast(loggerEnumItem, logLevel))                                                  \
            {                                                                                                          \
                hl_logger::log(loggerEnumItem,                                                                         \
                               logLevel,                                                                               \
                               forcePrintFileLine,                                                                     \
                               filename,                                                                               \
                               line,                                                                                   \
                               fmtMsg HLLOG_APPLY_WITH_LEADING_COMMA(HLLOG_DUPLICATE_PARAM, ##__VA_ARGS__));           \
            }                                                                                                          \
            HLLOG_TYPED_LAZY(loggerEnumItem, logLevel, fmtMsg, ##__VA_ARGS__);                                         \
        }                                                                                                              \
    }while(false)
#define HLLOG_TYPED_FULL(loggerEnumItem, logLevel, forcePrintFileLine, filename, line, fmtMsg, ...) \
    HLLOG_TYPED_FULL_PREFIXED(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, logLevel, forcePrintFileLine, filename, line, fmtMsg, ##__VA_ARGS__)


/**
 * @brief log message with parameters into a logger with a log level
 *
 * @param loggerEnumItem user logger enum item
 * @param loglevel log level
 * @param fmtMsg   fmt-lib format message
 * @param ...      parameters of the message
 *
 * @code
 * enum class UserLoggers{LOGGER1, LOGGER2, LOG_MAX};
 * ...
 * HLLOG_TYPED(LOGGER1, "hello {}", "world");
 * @endcode
 */
#define HLLOG_TYPED_PREFIXED(loggerEnumItem, logLevel, fmtMsg, ...) \
    HLLOG_TYPED_FULL_PREFIXED(loggerEnumItem, logLevel, false, HLLOG_FILENAME, __LINE__, FMT_COMPILE(fmtMsg), ##__VA_ARGS__)

#define HLLOG_TYPED(loggerEnumItem, logLevel, fmtMsg, ...) \
    HLLOG_TYPED_PREFIXED(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, logLevel, fmtMsg, ##__VA_ARGS__)

#ifdef HLLOG_ENABLE_LAZY_LOGGING
    #define HHLOG_UNTYPED_LAZY(logger, logLevel, fmtMsg, ...)                                                          \
        if (HLLOG_UNLIKELY(getLazyLoggingLevel(logger) <= logLevel))                                                   \
        {                                                                                                              \
            hl_logger::logLazy(logger,                                                                                 \
                               logLevel,                                                                               \
                               fmtMsg HLLOG_APPLY_WITH_LEADING_COMMA(HLLOG_DUPLICATE_PARAM, ##__VA_ARGS__));           \
       }
#else
    #define HHLOG_UNTYPED_LAZY(logger, logLevel, fmtMsg, ...)
#endif
/**
 * @brief log into a logger  with all the parameters. with or without fmtMsg compilation
 * @param loggerEnumItem     logger enum item
 * @param logLevel           message log level
 * @param forcePrintFileLine print source code file and line
 * @param filename           source code filename
 * @param line               source code line
 * @param fmtMsg             fmt-lib format message. must be FMT_COMPILE(fmtMsg) for compile. or fmtMsg if no compilation needed
 */
#define HLLOG_UNTYPED_FULL(logger, logLevel, forcePrintFileLine, filename, line, fmtMsg, ...)                          \
    do{                                                                                                                \
        if (HLLOG_UNLIKELY(getLoggingLevel(logger) <= logLevel))                                                       \
        {                                                                                                              \
            hl_logger::log(logger,                                                                                     \
                           logLevel,                                                                                   \
                           forcePrintFileLine,                                                                         \
                           filename,                                                                                   \
                           line,                                                                                       \
                           fmtMsg HLLOG_APPLY_WITH_LEADING_COMMA(HLLOG_DUPLICATE_PARAM, ##__VA_ARGS__));               \
        }                                                                                                              \
        HHLOG_UNTYPED_LAZY(logger, logLevel, fmtMsg, ##__VA_ARGS__);                                                   \
    }while(false)

/**
 * @brief log message with parameters into a logger with a log level
 *
 * @param logger   logger
 * @param loglevel log level
 * @param fmtMsg   fmt-lib format message
 * @param ...      parameters of the message
 *
 * @code
 * enum class UserLoggers{LOGGER1, LOGGER2, LOG_MAX};
 * ...
 * auto logger = logLogger(UserLoggers::LOGGER1);
 * HLLOG_UNTYPED(logger, "hello {}", "world");
 * @endcode
 */
#define HLLOG_UNTYPED(logger, logLevel, fmtMsg, ...) \
    HLLOG_UNTYPED_FULL(logger, logLevel, false, HLLOG_FILENAME, __LINE__, FMT_COMPILE(fmtMsg), ##__VA_ARGS__)

/**
 * @brief log message with parameters into a logger with a specific log level
 *
 * @param loggerEnumItem user logger enum item
 * @param fmtMsg  fmt-lib format message
 * @param ...     parameters of the message
 *
 *
 * @code
 * enum class UserLoggers{LOGGER1, LOGGER2, LOG_MAX};
 * ...
 * HLLOG_TRACE(LOGGER1, "hello {}", "world");
 * @endcode
 */
#define HLLOG_TRACE(loggerEnumItem, fmtMsg, ...)    HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_TRACE, fmtMsg, ##__VA_ARGS__)
#define HLLOG_DEBUG(loggerEnumItem, fmtMsg, ...)    HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_DEBUG, fmtMsg, ##__VA_ARGS__)
#define HLLOG_INFO(loggerEnumItem, fmtMsg, ...)     HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_INFO,  fmtMsg, ##__VA_ARGS__)
#define HLLOG_WARN(loggerEnumItem, fmtMsg, ...)     HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_WARN,  fmtMsg, ##__VA_ARGS__)
#define HLLOG_ERR(loggerEnumItem, fmtMsg, ...)      HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_ERROR, fmtMsg, ##__VA_ARGS__)
#define HLLOG_CRITICAL(loggerEnumItem, fmtMsg, ...) HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_CRITICAL, fmtMsg, ##__VA_ARGS__)

#define HLLOG_BY_LEVEL(loggerEnumItem, level, fmtMsg, ...) HLLOG_TYPED(logname, level, fmtMsg,  ## __VA_ARGS__)
/**
 * @brief log message with the current function name into a logger with a specific log level
 *
 * @param loggerEnumItem user logger enum item
 * @param fmtMsg  fmt-lib format message
 * @param ... parameters of the message
 *
 *
 * @code
 * enum class UserLoggers{LOGGER1, LOGGER2, LOG_MAX};
 * ...
 * HLLOG_TRACE_F(LOGGER1, "hello {}", "world");
 * @endcode
 */
#define HLLOG_TRACE_F(loggerEnumItem, fmtMsg, ...)    HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_TRACE, "{}: " fmtMsg, HLLOG_FUNC, ##__VA_ARGS__)
#define HLLOG_DEBUG_F(loggerEnumItem, fmtMsg, ...)    HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_DEBUG, "{}: " fmtMsg, HLLOG_FUNC, ##__VA_ARGS__)
#define HLLOG_INFO_F(loggerEnumItem, fmtMsg, ...)     HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_INFO,  "{}: " fmtMsg, HLLOG_FUNC, ##__VA_ARGS__)
#define HLLOG_WARN_F(loggerEnumItem, fmtMsg, ...)     HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_WARN,  "{}: " fmtMsg, HLLOG_FUNC, ##__VA_ARGS__)
#define HLLOG_ERR_F(loggerEnumItem, fmtMsg, ...)      HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_ERROR, "{}: " fmtMsg, HLLOG_FUNC, ##__VA_ARGS__)
#define HLLOG_CRITICAL_F(loggerEnumItem, fmtMsg, ...) HLLOG_TYPED(loggerEnumItem, HLLOG_LEVEL_CRITICAL, "{}: " fmtMsg, HLLOG_FUNC, ##__VA_ARGS__)

#define HLLOG_BY_LEVEL_F(loggerEnumItem, level, fmtMsg, ...) HLLOG_TYPED(loggerEnumItem, level, "{}: " fmtMsg, __FUNCTION__, ##__VA_ARGS__)

#define HLLOG_LEVEL_AT_LEAST_TRACE(loggerEnumItem)    (hl_logger::logLevelAtLeast(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, HLLOG_LEVEL_TRACE))
#define HLLOG_LEVEL_AT_LEAST_DEBUG(loggerEnumItem)    (hl_logger::logLevelAtLeast(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, HLLOG_LEVEL_DEBUG))
#define HLLOG_LEVEL_AT_LEAST_INFO(loggerEnumItem)     (hl_logger::logLevelAtLeast(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, HLLOG_LEVEL_INFO))
#define HLLOG_LEVEL_AT_LEAST_WARN(loggerEnumItem)     (hl_logger::logLevelAtLeast(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, HLLOG_LEVEL_WARN))
#define HLLOG_LEVEL_AT_LEAST_ERR(loggerEnumItem)      (hl_logger::logLevelAtLeast(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, HLLOG_LEVEL_ERROR))
#define HLLOG_LEVEL_AT_LEAST_CRITICAL(loggerEnumItem) (hl_logger::logLevelAtLeast(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, HLLOG_LEVEL_CRITICAL))

/**
 *  @brief set logging level for a logger
 *  @param loggerEnumItem logger enum item
 *  @param newLevel       new logging level
 */
#define HLLOG_SET_LOGGING_LEVEL(loggerEnumItem, newLevel) hl_logger::setLoggingLevel(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, newLevel)

/**
 *  @brief set logging level for a lazy logger
 *  @param loggerEnumItem logger enum item
 *  @param newLevel       new lazy logging level
 */
#define HLLOG_SET_LAZY_LOGGING_LEVEL(loggerEnumItem, newLevel) hl_logger::setLazyLoggingLevel(HLLOG_ENUM_TYPE_NAME::loggerEnumItem, newLevel)

/**
 * drop a specific logger
 * @param loggerEnumItem logger enum item
 * @return true if the logger was dropped successfully
 *         if the logger was already dropped - return false
 */
#define HLLOG_DROP(loggerEnumItem) hl_logger::drop(HLLOG_ENUM_TYPE_NAME::loggerEnumItem)

/**
 * @brief HLLOG_DEFINE_MODULE_LOGGER define module-specific data structures
 * @parameters a list of all the items of user-provided logger enum
 *
 * @code
 * enum class UserLoggers{LOGGER1, LOGGER2, LOG_MAX};
 * ...
 * HLLOG_DEFINE_MODULE_LOGGER(LOGGER1, LOGGER2, LOG_MAX)
 * @endcode
 */
#define HLLOG_DEFINE_MODULE_LOGGER(loggerEnumItem0, loggerEnumItem1, ...) \
                                   HLLOG_DEFINE_MODULE_LOGGER_IMPL(HLLOG_EMPTY(), loggerEnumItem0, loggerEnumItem1, ##__VA_ARGS__)

/**
 * @brief HLLOG_DECLARE_MODULE_LOGGER declare module-specific data structures
 * must be used after HLLOG_ENUM_TYPE_NAME definition and out of any namespace
 * @code
 * #define HLLOG_ENUM_TYPE_NAME MyNamespace::MyEnum
 * HLLOG_DECLARE_MODULE_LOGGER()
 * @endcode
 */
#define HLLOG_DECLARE_MODULE_LOGGER() HLLOG_DECLARE_MODULE_LOGGER_IMPL()

/**
 * HLLOG_DEFINE_MODULE_LOGGER with a private/public moduleLoggerData symbol visibility
 * if your module makes all the symbols private but this module is used by other modules for logging
 * you should make moduleLoggerData visible to other modules.
 * in this case use HLLOG_DEFINE_MODULE_LOGGER_PUBLIC
 */
#define HLLOG_DEFINE_MODULE_LOGGER_PUBLIC(loggerEnumItem0, loggerEnumItem1, ...) \
                                  HLLOG_DEFINE_MODULE_LOGGER_IMPL(HLLOG_API, loggerEnumItem0, loggerEnumItem1, ##__VA_ARGS__)
#define HLLOG_PRIVATE __attribute__((visibility("hidden")))
#define HLLOG_DEFINE_MODULE_LOGGER_PRIVATE(loggerEnumItem0, loggerEnumItem1, ...) \
                                  HLLOG_DEFINE_MODULE_LOGGER_IMPL(HLLOG_PRIVATE, loggerEnumItem0, loggerEnumItem1, ##__VA_ARGS__)

