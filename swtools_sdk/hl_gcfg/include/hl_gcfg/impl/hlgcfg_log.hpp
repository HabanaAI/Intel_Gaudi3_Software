#pragma once
#include <string_view>
#include <hl_logger/hllog_core.hpp>
#include <hl_logger/impl/hllog_macros.hpp>
#include <hl_logger/impl/hllog_fmt_headers.hpp>

#include "../hlgcfg_defs.hpp"

namespace hl_gcfg {
HLGCFG_NAMESPACE{
    hl_logger::LoggerSPtr getLogger();
    int getLoggingLevel();
}

HLGCFG_INLINE_NAMESPACE{
template<typename TFmtMsg, typename... Args>
inline void log(hl_logger::LoggerSPtr const & logger, int logLevel, std::string_view file, int line, TFmtMsg fmtMsg, Args&&... args)
try
{
    fmt::memory_buffer buf;
    fmt::format_to(std::back_inserter(buf), fmtMsg, std::forward<Args>(args)...);
    hl_logger::log(logger, logLevel, std::string_view(buf.data(), buf.size()), file, line, false);
}
catch(std::exception const & e)
{
    hl_logger::log(logger, HLLOG_LEVEL_ERROR, "failed to format message");
    hl_logger::log(logger, HLLOG_LEVEL_ERROR, e.what());
    hl_logger::logStackTrace(logger, HLLOG_LEVEL_ERROR);
}
catch(...)
{
    hl_logger::log(logger, HLLOG_LEVEL_ERROR, "failed to format message");
    hl_logger::logStackTrace(logger, HLLOG_LEVEL_ERROR);
}
}}

#define HLGCFG_UNTYPED(logLevel, fmtMsg, ...)                                                                          \
    do{                                                                                                                \
        if (hl_gcfg::getLoggingLevel() <= logLevel)                                                                    \
        {                                                                                                              \
            hl_gcfg::log(hl_gcfg::getLogger(),                                                                         \
                         logLevel,                                                                                     \
                         std::string_view(__FILE__),                                                                   \
                         __LINE__,                                                                                     \
                         FMT_COMPILE(fmtMsg),                                                                          \
                         ##__VA_ARGS__);                                                                               \
        }                                                                                                              \
    }while(false)

#define HLGCFG_LOG(level, fmtMsg, ...)   HLGCFG_UNTYPED(level, "{}: " fmtMsg, __func__, ##__VA_ARGS__)

#define HLGCFG_LOG_TRACE(fmtMsg, ...)    HLGCFG_LOG(HLLOG_LEVEL_TRACE, level, fmtMsg, ##__VA_ARGS__)
#define HLGCFG_LOG_DEBUG(fmtMsg, ...)    HLGCFG_LOG(HLLOG_LEVEL_DEBUG, fmtMsg, ##__VA_ARGS__)
#define HLGCFG_LOG_INFO(fmtMsg, ...)     HLGCFG_LOG(HLLOG_LEVEL_INFO,  fmtMsg, ##__VA_ARGS__)
#define HLGCFG_LOG_WARN(fmtMsg, ...)     HLGCFG_LOG(HLLOG_LEVEL_WARN,  fmtMsg, ##__VA_ARGS__)
#define HLGCFG_LOG_ERR(fmtMsg, ...)      HLGCFG_LOG(HLLOG_LEVEL_ERROR, fmtMsg, ##__VA_ARGS__)
#define HLGCFG_LOG_CRITICAL(fmtMsg, ...) HLGCFG_LOG(HLLOG_LEVEL_CRITICAL, fmtMsg, ##__VA_ARGS__)

