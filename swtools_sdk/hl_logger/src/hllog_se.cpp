// disable inline implementation to avoid collisions of different versions of fmt
#define HLLOG_DISABLE_INLINE_IMPLEMENTATION
#include "hl_logger/hllog_se.hpp"
#include "hllog_logger.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"

inline namespace HLLOG_SE_INLINE_NAMESPACE{
hl_logger::LoggerSPtr validate(std::string const& logname, std::string const& msg)
{
    auto log = hl_logger::getRegisteredLogger(logname);
    if (log == nullptr)
    {
        log = hl_logger::getRegisteredLogger("default");
        if (log == nullptr)
        {
            log =
                std::make_shared<hl_logger::Logger>("default", std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            spdlog::register_logger(log);
        }
        log->critical("logger with name - {}, does not exist [on msg \"{}\"]", logname, msg);
        return log;
    }
    else
    {
        return log;
    }
}

int getLogLevel(std::string const& logname)
{
    auto log = validate(logname, "getLogLevel");
    return log->level();
}

void SET_LOGGER_SINK(std::string const& logname, std::string const& pathname, int lvl, size_t size, size_t amount)
{
    auto logger = validate(logname, "set sink level");
    hl_logger::addFileSink(logger, pathname, size, amount, lvl);
}

void CREATE_SINK_LOGGER(std::string const& logName,
                        std::string const& fileName,
                        unsigned           defaultLogFileSize,
                        unsigned           logFileAmount,
                        unsigned           defaultLogLevel /*= HLLOG_LEVEL_CRITICAL */,
                        unsigned           sink /* = SINK_STDOUT*/)
{
    hl_logger::LoggerCreateParams params;
    params.logFileName     = fileName;
    params.logFileAmount    = logFileAmount;
    params.registerLogger   = true;
    params.consoleStream    = hl_logger::LoggerCreateParams::ConsoleStream(sink);
    params.logFileSize      = defaultLogFileSize;
    params.loggerFlushLevel = HLLOG_LEVEL_TRACE;

    auto log      = hl_logger::createLogger(logName, params);
    int  logLevel = hl_logger::getDefaultLoggingLevel(logName, defaultLogLevel);
    hl_logger::setLoggingLevel(log, logLevel);
}

void CREATE_LOGGER(std::string const& logName,
                   std::string const& fileName,
                   unsigned           defaultLogFileSize,
                   unsigned           logFileAmount,
                   unsigned           defaultLogLevel /*= HLLOG_LEVEL_CRITICAL */)
{
    CREATE_SINK_LOGGER(logName, fileName, defaultLogFileSize, logFileAmount, defaultLogLevel, SINK_STDOUT);
}

void CREATE_ERR_LOGGER(std::string const& logName,
                       std::string const& fileName,
                       unsigned           defaultLogFileSize,
                       unsigned           logFileAmount,
                       unsigned           defaultLogLevel /*= HLLOG_LEVEL_CRITICAL */)
{
    CREATE_SINK_LOGGER(logName, fileName, defaultLogFileSize, logFileAmount, defaultLogLevel, SINK_STDERR);
}
}  // namespace HLLOG_INLINE_API_NAMESPACE