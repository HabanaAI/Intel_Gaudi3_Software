#include <algorithm>
#include <cstdlib>
#include <deque>
#include <map>
#include "syn_logging.h"
#include "synapse_types.h"
#include "log_manager.h"
#include "dfa_defines.hpp"
#include "hcl_api.hpp"

#define LOGGER_NAME_MAX_LENGTH 15

namespace synapse
{
std::function<bool(int signal, const char* signalStr, bool isSevere, int stage)> m_onCrash;

LogManager& LogManager::instance()
{
    static LogManager instance;

    return instance;
}

static unsigned getEnvVarValue(const char * envvarName, unsigned defaultValue, bool allowSmallerValues)
{
    unsigned value = defaultValue;
    char* newValueStr = getenv(envvarName);
    if (newValueStr && (allowSmallerValues || atoi(newValueStr) > defaultValue))
    {
        value = atoi(newValueStr);
    }
    return value;
}
const unsigned synapseLogFileSize     = getEnvVarValue("SYNAPSE_LOG_FILE_SIZE", LOG_SIZE, false);
const unsigned synapseLogFileAmount   = getEnvVarValue("SYNAPSE_LOG_FILE_AMOUNT", LOG_AMOUNT, true);
const unsigned synapseRTLogFileSize   = getEnvVarValue("SYNAPSE_RT_LOG_FILE_SIZE", LOG_SIZE_RT, false);
const unsigned synapseRTLogFileAmount = getEnvVarValue("SYNAPSE_RT_LOG_FILE_AMOUNT", LOG_AMOUNT_RT, true);

static void createModuleLoggers(LogManager::LogType)
{
    hl_logger::LoggerCreateParams gcParams;
    gcParams.logFileName = SYNAPSE_LOG_SINK_FILE;
    gcParams.logFileSize = synapseLogFileSize;
    gcParams.logFileAmount = synapseLogFileAmount;
    gcParams.logFileBufferSize = 1024*1024;
    gcParams.separateLogFile = GRAPH_COMPILER_SEPARATE_LOG_FILE;
    gcParams.sepLogPerThread = true;
    gcParams.printSpecialContext = true;
    hl_logger::createLoggers({LogManager::LogType::GC,
                              LogManager::LogType::TRANSPOSE_SPLIT,
                              LogManager::LogType::VALIDATION,
                              LogManager::LogType::GRAPH_DATA,
                              LogManager::LogType::SLICE_NORM,
                              LogManager::LogType::SYNC_SCHEME,
                              LogManager::LogType::SYNC_SCHEME_DLT,
                              LogManager::LogType::RANGE_SLICE,
                              LogManager::LogType::FUSE_BATCH_NORM,
                              LogManager::LogType::SRAM_SLICE,
                              LogManager::LogType::BROADCAST_NODE_CREATOR,
                              LogManager::LogType::SLICE_NODE,
                              LogManager::LogType::MEM_COHERENCE,
                              LogManager::LogType::CSE_OPTIMIZATION,
                              LogManager::LogType::OP_VALIDATOR,
                              LogManager::LogType::GC_SHARED_LAYER,
                              LogManager::LogType::OPTIMIZE_SI,
                              LogManager::LogType::BGEMM_FLATTEN,
                              LogManager::LogType::SPILL_FILL,
                              LogManager::LogType::HUGE_TENSOR_SLICE,
                              LogManager::LogType::SRAM_SOL_GEN,
                              LogManager::LogType::LAYERED_BRAIN,
                              LogManager::LogType::LB_BUNDLER,
                              LogManager::LogType::LB_SLICER,
                              LogManager::LogType::LB_SCHEDULER,
                              LogManager::LogType::LB_CACHE_MNGR,
                              LogManager::LogType::LB_EVALUATOR,
                              LogManager::LogType::LB_PARTIALS,
                              LogManager::LogType::TILE_SIZE_CALC,
                              LogManager::LogType::BE_SLICER,
                              LogManager::LogType::COST_MODEL,
                              LogManager::LogType::BP_GRAPH,
                              LogManager::LogType::SYNC_SCHEME_VAL,
                              LogManager::LogType::QMAN,
                              LogManager::LogType::ROI_RANGE,
                              LogManager::LogType::MME_STACK,
                              LogManager::LogType::GRAD_PAIR,
                              LogManager::LogType::MEMORY_SECTION,
                              LogManager::LogType::HABANA_NODE,
                              LogManager::LogType::KERNEL_DB,
                              LogManager::LogType::RECIPE_GEN,
                              LogManager::LogType::ROI_SPLITTER,
                              LogManager::LogType::DCORE_SPLITTER,
                              LogManager::LogType::TPC_NODE,
                              LogManager::LogType::TPC_SLICE,
                              LogManager::LogType::OP_SLICE,
                              LogManager::LogType::OPT_LOGICAL_OPS,
                              LogManager::LogType::CTRL_LOGICAL_OP,
                              LogManager::LogType::MME_DESC_CACHE,
                              LogManager::LogType::SYN_COMPARE,
                              LogManager::LogType::SYNREC,
                              LogManager::LogType::SFG,
                              LogManager::LogType::OPT_MEMCPY,
                              LogManager::LogType::EAGER,
                              LogManager::LogType::GC_CONF,
                              LogManager::LogType::DMA_RANGE,
                              LogManager::LogType::LIVA,
                              LogManager::LogType::EPOCH_ALLOC,
                              LogManager::LogType::HEAP_ALLOC,
                              LogManager::LogType::TENSORS_ALLOC,
                              LogManager::LogType::PASS_MANAGER,
                              LogManager::LogType::FUNCTION_SCOPE,
                              LogManager::LogType::BIG_IMAGE_ALG,
                              LogManager::LogType::SPILL_RESIDUALS,
                              LogManager::LogType::DYN_SHAPE,
                              LogManager::LogType::SCHEDULER,
                              LogManager::LogType::QUANT,
                              LogManager::LogType::GC_TPC_FUSER,
                              LogManager::LogType::DATA_TYPES,
                              LogManager::LogType::GC_ARC,
                              LogManager::LogType::GC_COMPLEX_GUID,
                              LogManager::LogType::ZST_REMOVER,
                              LogManager::LogType::BASE_REGS_CACHE,
                              LogManager::LogType::GC_TRANSLATION,
                              LogManager::LogType::DATA_LAYOUT,
                              LogManager::LogType::CACHE_MAINT,
                              LogManager::LogType::CONST_FOLDING,
                              LogManager::LogType::FLASH_ATTENTION,
                              LogManager::LogType::FUSE_BROADCAST},
                             gcParams);

    hl_logger::LoggerCreateParams rtParams;
    rtParams.logFileName = SYNAPSE_LOG_SINK_FILE_RT;
    rtParams.logFileSize = synapseRTLogFileSize;
    rtParams.logFileAmount = synapseRTLogFileAmount;
    rtParams.logFileBufferSize = 1024*1024;
    rtParams.printSpecialContext = true;
    hl_logger::createLoggers({LogManager::LogType::SYN_STREAM,
                              LogManager::LogType::SYN_CS,
                              LogManager::LogType::SYN_PROG_DWNLD,
                              LogManager::LogType::SYN_MEM_ALLOC,
                              LogManager::LogType::SYN_DATA_CHUNK,
                              LogManager::LogType::SYN_RECIPE,
                              LogManager::LogType::SYN_RCPE_CACHE,
                              LogManager::LogType::SYN_PATCHING,
                              LogManager::LogType::SYN_GRAPH,
                              LogManager::LogType::SYN_DEVICE,
                              LogManager::LogType::SYN_MEM_MAP,
                              LogManager::LogType::SYN_WORK_COMPL,
                              LogManager::LogType::SYN_OSAL,
                              LogManager::LogType::SYN_TPC_PRINT,
                              LogManager::LogType::SYN_PATCH_INFO,
                              LogManager::LogType::SYN_EVENT_FD,
                              LogManager::LogType::SYN_DM_STREAM,
        }, rtParams);

    hl_logger::LoggerCreateParams synRtTest = rtParams;
    synRtTest.defaultLoggingLevel      = HLLOG_LEVEL_INFO;
    synRtTest.forceDefaultLoggingLevel = true;

    hl_logger::createLoggers({LogManager::LogType::SYN_RT_TEST
        }, synRtTest);

    rtParams.defaultLazyLoggingLevel = HLLOG_LEVEL_DEBUG;
    hl_logger::createLogger(LogManager::LogType::SYN_API, rtParams);
    // SYN_PROGRESS and SYN_DFA_API loggers are a lazy loggers (by default) used to log the longSo progress
    // and the user api-s
    {
        hl_logger::LoggerCreateParams workProgressParams = rtParams;

        workProgressParams.forceDefaultLazyLoggingLevel = true;
        workProgressParams.defaultLazyLoggingLevel      = HLLOG_LEVEL_TRACE;
        workProgressParams.defaultLoggingLevel          = HLLOG_LEVEL_OFF;
        workProgressParams.defaultLazyQueueSize         = HLLOG_DEFAULT_LAZY_QUEUE_SIZE * 2; // important logger, we want more

        hl_logger::createLoggers({LogManager::LogType::SYN_PROGRESS,
                                  LogManager::LogType::SYN_DFA_API
                                 }, workProgressParams);
    }

    hl_logger::LoggerCreateParams gcPerfParams;
    gcPerfParams.logFileName         = GRAPH_COMPILER_PERF_SEPARATE_LOG_FILE;
    gcPerfParams.logFileSize         = synapseLogFileSize;
    gcPerfParams.logFileAmount       = synapseLogFileAmount;
    gcPerfParams.logFileBufferSize   = 1024 * 1024;
    gcPerfParams.sepLogPerThread     = true;
    gcPerfParams.printThreadID       = false;
    gcPerfParams.printTime           = false;
    gcPerfParams.printLoggerName     = false;
    gcPerfParams.loggerFlushLevel    = HLLOG_LEVEL_ERROR;
    gcPerfParams.logLevelStyle       = hl_logger::LoggerCreateParams::LogLevelStyle::off;
    hl_logger::createLogger(LogManager::LogType::GC_PERF, gcPerfParams);
}

// on-demand loggers
static void createModuleLoggersOnDemand(LogManager::LogType)
{
    hl_logger::LoggerCreateParams params;
    params.logFileName = SYNAPSE_CS_PARSER_SEPARATE_LOG_FILE;
    params.logFileSize = synapseRTLogFileSize;
    params.logFileAmount = synapseRTLogFileAmount;
    params.logFileBufferSize = 1024*1024;
    hl_logger::createLoggerOnDemand(LogManager::LogType::SYN_CS_PARSER, params);

    {
        // SYN_DEV_FAIL logger is the main dfa logger
        hl_logger::LoggerCreateParams dfaParams;
        dfaParams.logFileName                  = DEVICE_FAIL_ANALYSIS_FILE;
        dfaParams.logFileSize                  = DEVICE_FAIL_ANALYSIS_FILE_SIZE;
        dfaParams.logFileAmount                = 0;
        dfaParams.logFileBufferSize            = 64*1024; // 64KB to make sure it's enough for any long message
        dfaParams.defaultLoggingLevel          = HLLOG_LEVEL_TRACE;
        dfaParams.forceDefaultLoggingLevel     = true;
        dfaParams.defaultLazyLoggingLevel      = HLLOG_LEVEL_OFF;
        dfaParams.forceDefaultLazyLoggingLevel = true;
        dfaParams.loggerFlushLevel             = HLLOG_LEVEL_TRACE;
        dfaParams.consoleStream                = hl_logger::LoggerCreateParams::ConsoleStream::disabled;
        dfaParams.rotateLogfileOnOpen          = false;
        hl_logger::createLoggerOnDemand(LogManager::LogType::SYN_DEV_FAIL, dfaParams);

        // SYN_FAIL_RECIPE is used during dfa to log in-flight recipes information
        dfaParams.logFileName   = SUSPECTED_RECIPES;
        dfaParams.logFileSize   = SUSPECTED_RECIPES_FILE_SIZE;
        dfaParams.logFileAmount = 0;
        dfaParams.consoleStream = hl_logger::LoggerCreateParams::ConsoleStream::disabled;
        hl_logger::createLoggerOnDemand(LogManager::LogType::SYN_FAIL_RECIPE, dfaParams);

        // DMESG_LOG is used during dfa to log a copy of the dmesg
        dfaParams.logFileName   = DMESG_COPY_FILE;
        dfaParams.logFileSize   = DMESG_COPY_FILE_SIZE;
        dfaParams.logFileAmount = 0;
        dfaParams.consoleStream = hl_logger::LoggerCreateParams::ConsoleStream::disabled;
        hl_logger::createLoggerOnDemand(LogManager::LogType::DMESG_LOG, dfaParams);

        // DFA_NIC logger is used to log nic specific information during DFA
        dfaParams.logFileName   = DFA_NIC_INFO_FILE;
        dfaParams.logFileSize   = DFA_NIC_INFO_SIZE;
        dfaParams.logFileAmount = 0;
        dfaParams.consoleStream = hl_logger::LoggerCreateParams::ConsoleStream::disabled;
        hl_logger::createLoggerOnDemand(LogManager::LogType::DFA_NIC, dfaParams);
    }

    // DFA_API_INFO is used to log headers (like dfa start time) to the DFA_API_FILE
    {
        hl_logger::LoggerCreateParams dfaApiInfo;
        dfaApiInfo.logFileName                  = DFA_API_FILE;
        dfaApiInfo.logFileSize                  = DFA_API_FILE_SIZE;
        dfaApiInfo.logFileAmount                = 0;
        dfaApiInfo.rotateLogfileOnOpen          = false;
        dfaApiInfo.defaultLoggingLevel          = HLLOG_LEVEL_TRACE;
        dfaApiInfo.forceDefaultLoggingLevel     = true;
        dfaApiInfo.defaultLazyLoggingLevel      = HLLOG_LEVEL_OFF;
        dfaApiInfo.forceDefaultLazyLoggingLevel = true;

        hl_logger::createLoggerOnDemand(LogManager::LogType::DFA_API_INFO, dfaApiInfo);
    }

    {
        hl_logger::LoggerCreateParams etlParams;
        etlParams.logFileName       = EVENT_TRIGGER_SEPARATE_LOG_FILE;
        etlParams.logFileSize       = synapseLogFileSize;
        etlParams.logFileAmount     = synapseLogFileAmount;
        etlParams.logFileBufferSize = 1024 * 1204;
        hl_logger::createLoggerOnDemand(LogManager::LogType::EVENT_TRIGGER, etlParams);
    }

    {
        hl_logger::LoggerCreateParams perfParams;
        perfParams.logFileName       = PERFORMANCE_MEASURMENTS_COLLECT_FILE;
        perfParams.logFileSize       = PERFORMANCE_LOG_SIZE;
        perfParams.logFileAmount     = PERFORMANCE_LOG_AMOUNT;
        perfParams.logFileBufferSize = 1024 * 1204;
        hl_logger::createLoggerOnDemand(LogManager::LogType::PERF, perfParams);
    }

    {
        hl_logger::LoggerCreateParams statParams;
        statParams.logFileName   = RECIPE_STATS_COLLECT_FILE;
        statParams.logFileSize   = RECIPE_STATS_LOG_SIZE;
        statParams.logFileAmount = RECIPE_STATS_LOG_AMOUNT;
        hl_logger::createLoggerOnDemand(LogManager::LogType::RECIPE_STATS, statParams);
    }
}

static void onModuleLoggersCrashSignal(LogManager::LogType, int signal, const char* signalStr, bool isSevere)
{
    bool synInitialized = false;
    if (m_onCrash)
    {
        synInitialized = m_onCrash(signal, signalStr, isSevere, 0);
    }

    int level = isSevere ? HLLOG_LEVEL_ERROR : HLLOG_LEVEL_WARN;

    std::string errMsg = fmt::format("exception {} {}. severity: {}", signal, signalStr, isSevere ? "high" : "low");

    if (synInitialized)
    {
        HLLOG_TYPED(SYN_DEV_FAIL, level, "{}", errMsg);
    }
    // log into synapse_runtime log file as well
    HLLOG_TYPED(SYN_API, level, "{}", errMsg);

    if (!synInitialized)
    {
        return;
    }

    hl_logger::logStacktrace(LogManager::LogType::SYN_DEV_FAIL, level);

    if (m_onCrash)
    {
        m_onCrash(signal, signalStr, isSevere, 1);
    }
}

LogManager::LogManager()
{
}

void LogManager::enablePeriodicFlush(bool enable)
{
    hl_logger::enablePeriodicFlush(enable);
}

LogManager::~LogManager()
{
}

void LogManager::drop_log(const LogManager::LogType& logType)
{
    hl_logger::drop(logType);
}

void LogManager::set_log_level(const LogManager::LogType& logType, unsigned log_level)
{
    hl_logger::setLoggingLevel(logType, log_level);
}

// For testing: allows the caller to change the sink. Either to the given logSinks (if exists),
// if not, then to the given file
LogManager::LogSinks LogManager::setLogSinks(const LogManager::LogType& logType, LogSinks logSinks)
{
    hl_logger::flush(logType);
    return hl_logger::setSinks(logType, std::move(logSinks));
}

LogManager::LogSinks LogManager::getLogSinks(const LogManager::LogType& logType) const
{
    return hl_logger::getSinks(logType);
}

void LogManager::setLogSinks(const LogManager::LogType& logType, const std::string& newLogFileName)
{
    hl_logger::setSinks(logType);
    hl_logger::addFileSink(logType, newLogFileName, 2*1024*1024*1024ull, 0);
}

void LogManager::setOnCrash(std::function<bool(int signal, const char* signalStr, bool isSevere, int stage)> onCrash)
{
    m_onCrash = onCrash;
}

bool LogManager::getLogsFolderPath(std::string& logsFolderPath)
{
    logsFolderPath = hl_logger::getLogsFolderPath();
    return !logsFolderPath.empty();
}
void LogManager::flush()
{
    hl_logger::flushAll<LogType>();
}

void LogManager::set_logger_sink(const LogType& logType, const std::string& pathname, unsigned lvl, size_t size, size_t amount)
{
    hl_logger::addFileSink(logType, pathname, size, amount, lvl);
}

void LogManager::log_wrapper(const LogManager::LogType& logType, const int logLevel, std::string&& s)
{
    HLLOG_TYPED_PREFIXED(logType, logLevel, "{}", s);
}

void LogManager::setLogContext(const std::string& logContext)
{
    hl_logger::addCurThreadSpecialContext(logContext);
}

void LogManager::clearLogContext()
{
    hl_logger::removeCurThreadSpecialContext();
}

FuncScopeLog::FuncScopeLog(const std::string& function): m_function(function)
{
    LOG_TRACE(FUNCTION_SCOPE, "{} - function begin", m_function);
}

FuncScopeLog::~FuncScopeLog()
{
    LOG_TRACE(FUNCTION_SCOPE, "{} - function end", m_function);
}

}

HLLOG_DEFINE_MODULE_LOGGER(GC,
                           TRANSPOSE_SPLIT,
                           LIVA,
                           EPOCH_ALLOC,
                           HEAP_ALLOC,
                           TENSORS_ALLOC,
                           BE_SLICER,
                           SRAM_SLICE,
                           SRAM_SOL_GEN,
                           LAYERED_BRAIN,
                           LB_BUNDLER,
                           LB_SLICER,
                           LB_SCHEDULER,
                           LB_CACHE_MNGR,
                           LB_EVALUATOR,
                           LB_PARTIALS,
                           TILE_SIZE_CALC,
                           COST_MODEL,
                           BP_GRAPH,
                           SYN_API,
                           SYN_STREAM,
                           SYN_CS,
                           SYN_PROG_DWNLD,
                           SYN_MEM_ALLOC,
                           SYN_DATA_CHUNK,
                           SYN_RECIPE,
                           SYN_COMPARE,
                           SYN_RCPE_CACHE,
                           SYN_PATCHING,
                           SYN_GRAPH,
                           SYN_DEVICE,
                           SYN_MEM_MAP,
                           SYN_WORK_COMPL,
                           SYN_DEV_FAIL,
                           SYN_CS_PARSER,
                           SYN_PATCH_INFO,
                           SYN_EVENT_FD,
                           SYN_OSAL,
                           SYN_TPC_PRINT,
                           SYN_TEST,
                           RESNET_TEST,
                           MME_RUNNER,
                           GO_TEST,
                           TESTLOGGER,
                           HW_COVERAGE,
                           GRAPH_DATA,
                           SLICE_NORM,
                           SYNC_SCHEME,
                           SYNC_SCHEME_DLT,
                           RANGE_SLICE,
                           FUSE_BATCH_NORM,
                           SYNC_SCHEME_VAL,
                           ROI_RANGE,
                           MME_STACK,
                           GC_CONF,
                           DMA_RANGE,
                           EVENT_TRIGGER,
                           QMAN,
                           MEMORY_SECTION,
                           PERF,
                           RECIPE_STATS,
                           PASS_MANAGER,
                           FUNCTION_SCOPE,
                           HABANA_NODE,
                           KERNEL_DB,
                           RECIPE_GEN,
                           ROI_SPLITTER,
                           DCORE_SPLITTER,
                           TPC_NODE,
                           TPC_SLICE,
                           OP_SLICE,
                           OPT_LOGICAL_OPS,
                           CTRL_LOGICAL_OP,
                           MME_DESC_CACHE,
                           BIG_IMAGE_ALG,
                           SPILL_RESIDUALS,
                           DYN_SHAPE,
                           SCHEDULER,
                           QUANT,
                           VALIDATION,
                           GC_TPC_FUSER,
                           DATA_TYPES,
                           GC_ARC,
                           GC_COMPLEX_GUID,
                           SPILL_FILL,
                           HUGE_TENSOR_SLICE,
                           ZST_REMOVER,
                           BASE_REGS_CACHE,
                           BROADCAST_NODE_CREATOR,
                           SLICE_NODE,
                           MEM_COHERENCE,
                           CSE_OPTIMIZATION,
                           OP_VALIDATOR,
                           GC_SHARED_LAYER,
                           OPTIMIZE_SI,
                           BGEMM_FLATTEN,
                           GRAD_PAIR,
                           DMESG_LOG,
                           SYNREC,
                           GC_TRANSLATION,
                           SFG,
                           SYN_FAIL_RECIPE,
                           OPT_MEMCPY,
                           EAGER,
                           DATA_LAYOUT,
                           CACHE_MAINT,
                           SYN_DM_STREAM,
                           GC_PERF,
                           DFA_NIC,
                           SYN_PROGRESS,
                           SYN_DFA_API,
                           DFA_API_INFO,
                           SYN_RT_TEST,
                           CONST_FOLDING,
                           FLASH_ATTENTION,
                           FUSE_BROADCAST,
                           STRIDED_OP_DECODE,
                           LOG_MAX)
namespace synapse{
// functions use specialization of ModuleLoggerData that's why they must be after HLLOG_DEFINE_MODULE_LOGGER
const std::string_view LogManager::getLogTypeString(const LogType& logType) const
{
    return hl_logger::getLoggerEnumItemName(logType);
}

void LogManager::create_logger(const LogManager::LogType& logType,
                               const std::string&         fileName,
                               unsigned                   logFileSize,
                               unsigned                   logFileAmount,
                               const char*                separateLogFile,
                               bool                       sepLogPerThread)
{
    hl_logger::LoggerCreateParams params;
    params.logFileName      = fileName;
    params.logFileAmount    = logFileAmount;
    params.logFileSize      = logFileSize;
    params.separateLogFile  = separateLogFile ? separateLogFile : "";
    params.sepLogPerThread  = sepLogPerThread;
    params.loggerFlushLevel = HLLOG_LEVEL_WARN;
    params.defaultLoggingLevel = hl_logger::getLoggingLevel(logType) != HLLOG_LEVEL_OFF ?
                                          hl_logger::defaultLoggingLevel : HLLOG_LEVEL_ERROR;

    auto loggerName = hl_logger::getLoggerEnumItemName(logType);

    auto logger = hl_logger::getLogger(logType);
    if (logger != nullptr)
    {
        hl_logger::log(logger, HLLOG_LEVEL_CRITICAL, fmt::format("Logger was redefined {}", loggerName));
        return;
    }

    hl_logger::createLogger(logType, params);
}

}

DfaLoggersV3 getDfaLoggersV3()
{
    DfaLoggersV3 dfaLoggers {};

    dfaLoggers.dfaSynDevFailLogger   = hl_logger::getLogger(synapse::LogManager::LogType::SYN_DEV_FAIL);
    dfaLoggers.dfaDmesgLogger        = hl_logger::getLogger(synapse::LogManager::LogType::DMESG_LOG);
    dfaLoggers.dfaFailedRecipeLogger = hl_logger::getLogger(synapse::LogManager::LogType::SYN_FAIL_RECIPE);
    dfaLoggers.dfaNicInfoLogger      = hl_logger::getLogger(synapse::LogManager::LogType::DFA_NIC);
    dfaLoggers.dfaApi                = hl_logger::getLogger(synapse::LogManager::LogType::SYN_DFA_API);
    dfaLoggers.dfaApiInfo            = hl_logger::getLogger(synapse::LogManager::LogType::DFA_API_INFO);

    return dfaLoggers;
}

DfaLoggersV2 getDfaLoggersV2()
{
    DfaLoggersV2 dfaLoggers {};

    dfaLoggers.dfaSynDevFailLogger   = hl_logger::getLogger(synapse::LogManager::LogType::SYN_DEV_FAIL);
    dfaLoggers.dfaDmesgLogger        = hl_logger::getLogger(synapse::LogManager::LogType::DMESG_LOG);
    dfaLoggers.dfaFailedRecipeLogger = hl_logger::getLogger(synapse::LogManager::LogType::SYN_FAIL_RECIPE);
    dfaLoggers.dfaNicInfoLogger      = hl_logger::getLogger(synapse::LogManager::LogType::DFA_NIC);

    return dfaLoggers;
}
