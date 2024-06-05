#include "logger.hpp"
namespace hl_gcfg{
static void createModuleLoggers(LoggerTypes)
{
}

// on-demand loggers
static void createModuleLoggersOnDemand(LoggerTypes)
{
    hl_logger::LoggerCreateParams params;
    params.logFileName = "gcfg_log.txt";
    params.defaultLoggingLevel = HLLOG_LEVEL_WARN;
    params.printProcessID      = false;
    params.printThreadID       = true;
    params.logFileSize         = 1024*1024;
    params.logFileAmount       = 3;
    hl_logger::createLoggerOnDemand(LoggerTypes::HL_GCFG, params);
}
}  // namespace hl_gcfg

HLLOG_DEFINE_MODULE_LOGGER(HL_GCFG, LOG_MAX)
