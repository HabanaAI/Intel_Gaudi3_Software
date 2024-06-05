#ifdef SWTOOLS_DEP
#include "logger.h"

namespace mme_stack
{
namespace log
{
// create loggers (all the log files are created immediately when the module is loaded)
static void createModuleLoggers(Type)
{
    hl_logger::LoggerCreateParams params;
    params.logFileName = "mme_stack.log";
    params.logFileSize = 10 * 1024 * 1024;
    params.rotateLogfileOnOpen = true;
    params.logFileAmount = 10;
    params.logFileBufferSize = 1024 * 1024;
    params.sepLogPerThread = true;
    params.printSpecialContext = true;
    params.printThreadID = false;
    params.loggerFlushLevel = HLLOG_LEVEL_TRACE;
    hl_logger::createLogger(Type::MME_BRAIN, params);
    hl_logger::createLogger(Type::MME_CONFIG_PARSER, params);
    hl_logger::createLogger(Type::MME_RECIPE, params);

    hl_logger::LoggerCreateParams descDumpParams;
    descDumpParams.logFileName = "mme_desc_dump.log";
    descDumpParams.logFileSize = 10 * 1024 * 1024;
    descDumpParams.rotateLogfileOnOpen = true;
    descDumpParams.logFileAmount = 10;
    descDumpParams.logFileBufferSize = 1024 * 1024;
    descDumpParams.loggerFlushLevel = HLLOG_LEVEL_TRACE;
    descDumpParams.logLevelStyle = hl_logger::LoggerCreateParams::LogLevelStyle::off;
    descDumpParams.printThreadID       = false;
    descDumpParams.printTime           = false;
    descDumpParams.printLoggerName     = false;
    descDumpParams.printSpecialContext = false;
    descDumpParams.defaultLoggingLevel = HLLOG_LEVEL_INFO;
    hl_logger::createLogger(Type::MME_DESC_DUMP, descDumpParams);
}
}  // namespace log
}  // namespace mme_stack

HLLOG_DEFINE_MODULE_LOGGER(MME_BRAIN, MME_CONFIG_PARSER, MME_RECIPE, MME_DESC_DUMP, LOG_MAX);

#endif
