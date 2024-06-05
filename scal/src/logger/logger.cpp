#include "logger.h"
#include <cxxabi.h>
#include <execinfo.h>

static unsigned getEnvVarValue(const char * envvarName, unsigned defaultValue, bool allowSmallerValues)
{
    unsigned value       = defaultValue;
    char*    newValueStr = getenv(envvarName);

    if (newValueStr && (allowSmallerValues || atoi(newValueStr) > defaultValue))
    {
        value = atoi(newValueStr);
    }

    return value;
}

namespace scal{
// Same as RT log
#define LOG_AMOUNT_SCAL  5
const unsigned scalLogFileAmount = getEnvVarValue("SYNAPSE_RT_LOG_FILE_AMOUNT", LOG_AMOUNT_SCAL, false);

// create loggers (all the log files are created immediately)
static void createModuleLoggers(LoggerTypes)
{
    // one logger in one file with non-default parameters
    hl_logger::LoggerCreateParams params;
    params.logFileName   = "scal_log.txt";
    params.logFileAmount = scalLogFileAmount;
    params.logFileSize   = 10 * 1024 * 1024;
    params.defaultLoggingLevel = HLLOG_LEVEL_ERROR;
    hl_logger::createLogger(LoggerTypes::SCAL, params);
}

static void onModuleLoggersBeforeDestroy(LoggerTypes)
{
    LOG_INFO(SCAL, "closing logger. no more log messages will be logged");
}
}  // namespace YourNamespace

// define logger internal variables. requires a list of all the logger names (for string representation)
HLLOG_DEFINE_MODULE_LOGGER(SCAL, LOG_MAX)
