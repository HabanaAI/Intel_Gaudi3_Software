#include "synapse_logger.h"
namespace synapse{
//
static void createModuleLoggers(LoggerTypes)
{
    // todo: consider create logger file and attach a logger ?
    hl_logger::createLoggers({LoggerTypes::SCAL_1, LoggerTypes::SCAL_2}, {"log.txt"});
    hl_logger::LoggerCreateParams params;
    params.logFileName         = "scal_api_log.txt";
    params.defaultLoggingLevel = HLLOG_LEVEL_TRACE;
    params.printProcessID      = true;
    hl_logger::createLogger(LoggerTypes::SCAL_API, params);
    params.printTime           = false;
    hl_logger::createLogger(LoggerTypes::SCAL_API_3, params);
}

// all the following functions are optional and can be omitted (removed)

// on-demand loggers
static void createModuleLoggersOnDemand(LoggerTypes)
{
    hl_logger::LoggerCreateParams params;
    params.logFileName = "syn_api_log.txt";
    params.defaultLoggingLevel = HLLOG_LEVEL_TRACE;
    params.printProcessID      = false;
    params.printThreadID       = false;
    params.logFileBufferSize   = 1024*1024;
    params.logFileSize         = 1024*1024;
    params.logFileAmount       = 3;
    params.logLevelStyle       = hl_logger::LoggerCreateParams::LogLevelStyle::one_letter;
    hl_logger::createLoggersOnDemand({LoggerTypes::SYN_API, LoggerTypes::SYN_API2}, params);

    params.logFileName = "syn_api_err.txt";
    hl_logger::createLoggerOnDemand(LoggerTypes::SYN_API_CRASH, params);

    params.logFileName   = "dfa.txt";
    params.logFileAmount = 10;
    params.rotateLogfileOnOpen = true;
    hl_logger::createLoggerOnDemand(LoggerTypes::DFA, params);
}

static void onModuleLoggersBeforeDestroy(LoggerTypes)
{
    HLLOG_INFO(SYN_API, "closing synapse logger. no more log messages will be logged****");
}

static void onModuleLoggersCrashSignal(LoggerTypes, int signal, const char* signalStr, bool isSevere)
{
    HLLOG_ERR(SYN_API_CRASH, "crash. signal : {} {}. Severity: {}", signal, signalStr, isSevere ? "high" : "low");
    hl_logger::logStacktrace(LoggerTypes::SYN_API_CRASH, isSevere ? HLLOG_LEVEL_ERROR : HLLOG_LEVEL_INFO);
    HLLOG_INFO(SYN_API_CRASH, "closing synapse logger. no more log messages will be logged");
}
}  // namespace synapse

HLLOG_DEFINE_MODULE_LOGGER(SCAL_1,
                           SCAL_2,
                           SCAL_3,
                           SCAL_API,
                           SCAL_API_3,
                           SYN_API,
                           SYN_API2,
                           SYN_API_CRASH,
                           SYN_STREAM,
                           DFA,
                           SYN_TEST,
                           LOG_MAX)
