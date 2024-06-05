#pragma once
#include "hllog_core.hpp"

#if defined(FMT_VERSION) && FMT_VERSION != 50201
#error "fmt of an incompatible version was already included. only one version of fmt is allowed in one translation unit"
#endif

#define SINK_STDOUT 0
#define SINK_STDERR 1

#define HLLOG_SE_INLINE_NAMESPACE v1_0
#define TURN_ON_TRACE_MODE_LOGGING()  hl_logger::enableTraceMode(true)
#define TURN_OFF_TRACE_MODE_LOGGING() hl_logger::enableTraceMode(false)
inline namespace HLLOG_SE_INLINE_NAMESPACE {
HLLOG_API hl_logger::LoggerSPtr validate(std::string const& logname, std::string const& msg);

HLLOG_API int getLogLevel(std::string const& logname);

HLLOG_API void SET_LOGGER_SINK(std::string const& logname, std::string const& pathname, int level, size_t size, size_t amount);

HLLOG_API void DEFINE_LOGGER(std::string const& logname, std::string const& filename, unsigned size, unsigned amount, unsigned sink);

HLLOG_API void CREATE_SINK_LOGGER(std::string const& logName,
                                  std::string const& fileName,
                                  unsigned           defaultLogFileSize,
                                  unsigned           logFileAmount,
                                  unsigned           defaultLogLevel = HLLOG_LEVEL_CRITICAL,
                                  unsigned           sink            = SINK_STDOUT);

HLLOG_API void CREATE_LOGGER(std::string const& logName,
                             std::string const& fileName,
                             unsigned           defaultLogFileSize,
                             unsigned           logFileAmount,
                             unsigned           defaultLogLevel = HLLOG_LEVEL_CRITICAL);

HLLOG_API void CREATE_ERR_LOGGER(std::string const& logName,
                                 std::string const& fileName,
                                 unsigned           defaultLogFileSize,
                                 unsigned           logFileAmount,
                                 unsigned           defaultLogLevel = HLLOG_LEVEL_CRITICAL);
}  // namespace HLLOG_INLINE_API_NAMESPACE

HLLOG_BEGIN_NAMESPACE
template<typename... Args>
inline void log(LoggerSPtr const& logger,
                int               logLevel,
                bool              printFileLine,
                std::string_view  file,
                int               line,
                const char*       fmtMsg,
                Args const&... args);
HLLOG_END_NAMESPACE

#define GET_LOGGER(logname, fmtMsg) ::validate(logname, fmtMsg)

#define LOG_LVL(logname, level, fmtMsg, ...)                                                                           \
    hl_logger::log(GET_LOGGER(logname, "log"), level, false, std::string_view(), 0, fmtMsg, ##__VA_ARGS__)

#define LOG_TRACE(logname, fmtMsg, ...)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_LVL(logname, HLLOG_LEVEL_TRACE, fmtMsg, ##__VA_ARGS__);                                                    \
    } while (0)

#define LOG_DEBUG(logname, fmtMsg, ...)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_LVL(logname, HLLOG_LEVEL_DEBUG, fmtMsg, ##__VA_ARGS__);                                                    \
    } while (0)

#define LOG_INFO(logname, fmtMsg, ...)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_LVL(logname, HLLOG_LEVEL_INFO, fmtMsg, ##__VA_ARGS__);                                                     \
    } while (0)

#define LOG_WARN(logname, fmtMsg, ...)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_LVL(logname, HLLOG_LEVEL_WARN, fmtMsg, ##__VA_ARGS__);                                                     \
    } while (0)

#define LOG_ERR(logname, fmtMsg, ...)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_LVL(logname, HLLOG_LEVEL_ERROR, fmtMsg, ##__VA_ARGS__);                                                    \
    } while (0)

#define LOG_CRITICAL(logname, fmtMsg, ...)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_LVL(logname, HLLOG_LEVEL_CRITICAL, fmtMsg, ##__VA_ARGS__);                                                 \
    } while (0)

#define SET_LOG_LEVEL(logname, loglevel)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        auto logger = GET_LOGGER(logname, "set log level");                                                            \
        hl_logger::setLoggingLevel(logger, loglevel);                                                                  \
    } while (0)

#define FLUSH_LOGGER(logname, msg) hl_logger::flush(GET_LOGGER(logname, msg))
#define DROP_ALL_LOGGERS           hl_logger::dropAllRegisteredLoggers()
#define DROP_LOGGER(logname)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        FLUSH_LOGGER(logname, "drop logger");                                                                          \
        hl_logger::dropRegisteredLogger(logname);                                                                      \
    } while (0)

#ifndef HLLOG_DISABLE_INLINE_IMPLEMENTATION
#include <spdlog/include/spdlog/fmt/bundled/format.h>
#include <spdlog/include/spdlog/fmt/bundled/ostream.h>
#include "impl/hllog_se.inl"
#endif