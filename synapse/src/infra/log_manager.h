#pragma once
#define HLLOG_ENABLE_LAZY_LOGGING
#include <hl_logger/hllog.hpp>
#include <chrono>

#ifndef unlikely
#define unlikely HLLOG_UNLIKELY
#endif
#ifndef likely
#define likely HLLOG_LIKELY
#endif
namespace synapse
{
class LogManager
{
public:
    enum class LogType : uint32_t
    {
        GC,
        LIVA,
        EPOCH_ALLOC,
        HEAP_ALLOC,
        TENSORS_ALLOC,
        BE_SLICER,
        SRAM_SLICE,
        SRAM_SOL_GEN,
        TILE_SIZE_CALC,
        COST_MODEL,
        BP_GRAPH,
        SYN_API,         // API level                - synapse api, syn_singleton
        SYN_STREAM,      // stream level             - stream-related operations
        SYN_CS,          // command submission level - command submission, command buffer
        SYN_PROG_DWNLD,  // program download level   - download program data
        SYN_MEM_ALLOC,   // memory allocations level - device/host allocations, memory managers
        SYN_DATA_CHUNK,  // data chunks level        - data chunks allocations, cache, stats
        SYN_RECIPE,      // recipe level             - recipe parsing, handling
        SYN_COMPARE,     // recipe comparison level   - recipe comparison, handling
        SYN_RCPE_CACHE,  // recipe cache level       - RecipeCacheManager
        SYN_PATCHING,    // patching level           - recipe patching
        SYN_GRAPH,       // graph level              - compilation, handling
        SYN_DEVICE,      // device level             - device-specific functionality
        SYN_MEM_MAP,     // memory mapping level     - virtual-to-host / host-to-virtual
        SYN_WORK_COMPL,  // work completion level    - WorkCompletionManager
        SYN_DEV_FAIL,
        SYN_CS_PARSER,  // InflightCsParser
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
        GC_COMPLEX_GUID,  // used by GC internal complex guid extractor
        SPILL_FILL,
        HUGE_TENSOR_SLICE,
        ZST_REMOVER,
        BASE_REGS_CACHE,
        BROADCAST_NODE_CREATOR,
        SLICE_NODE,
        LAYERED_BRAIN,
        LB_EVALUATOR,   // Layered brain sub-component
        LB_BUNDLER,     // Layered brain sub-component
        LB_SLICER,      // Layered brain sub-component
        LB_SCHEDULER,   // Layered brain sub-component
        LB_CACHE_MNGR,  // Layered brain sub-component
        LB_PARTIALS,
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
        TRANSPOSE_SPLIT,
        DATA_LAYOUT,
        CACHE_MAINT,
        SYN_DM_STREAM,  // PDMA-Channel (Direct-Mode)
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
        LOG_MAX  // Must be last
    };

    static LogManager& instance();

    ~LogManager();

    void create_logger(const LogType& logType, const std::string& fileName,unsigned logFileSize, unsigned logFileAmount, const char* separateLogFile = nullptr, bool sepLogPerThread = false);

    void drop_log(const LogType& logType);

    void set_log_level(const LogType& logType, unsigned log_level);

    static unsigned get_log_level(const LogManager::LogType logType)
    {
        return hl_logger::getLoggingLevel(logType);
    }

    static unsigned get_log_level_no_check(const LogManager::LogType logType)
    {
        return hl_logger::getLoggingLevel(logType);
    }

    void set_logger_sink(const LogType& logType, const std::string& pathname, unsigned lvl, size_t size, size_t amount);

    void log_wrapper(const LogManager::LogType& logType, const int logLevel, std::string&& s);
    void setLogContext(const std::string& logContext);

    void clearLogContext();

    static bool getLogsFolderPath(std::string& logsFolderPath);

    void flush();

    // in some scenarios periodic flush causes issues.
    // so it's required to disable periodic flush in such cases
    // e.i. synDestroy, death tests etc.
    void enablePeriodicFlush(bool enable = true);
    using LogSinks = hl_logger::SinksSPtr;
    // getLogSinks() & setLogSinks() are used for testing, to change the log file and later recover it
    LogSinks setLogSinks(const LogManager::LogType& logType, LogSinks logSinks = LogSinks());
    LogSinks getLogSinks(const LogManager::LogType& logType) const;
    void     setLogSinks(const LogManager::LogType& logType, const std::string& newLogFileName);
    void     setOnCrash(std::function<bool(int signal, const char* signalStr, bool isSevere, int stage)> onCrash);
    const std::string_view getLogTypeString(const LogType& logType) const;

private:
    LogManager();
};

// Class for scoped log objects. The object will log at the creation and destruction execution only.
class FuncScopeLog
{
public:
    FuncScopeLog(const std::string& function);

    ~FuncScopeLog();

private:
    const std::string m_function;
};

}  // namespace synapse

#define HLLOG_ENUM_TYPE_NAME synapse::LogManager::LogType
HLLOG_DECLARE_MODULE_LOGGER()

using TempLogContextSetter = hl_logger::ScopedLogContext;

#define SET_TEMP_LOG_CONTEXT(context) TempLogContextSetter __tempLogContextSetter(context);

template<int LEVEL>
constexpr bool log_level_at_least(const synapse::LogManager::LogType logType)
{
    return hl_logger::logLevelAtLeast(logType, LEVEL);
}

template<int LEVEL, synapse::LogManager::LogType logType>
constexpr bool log_level_at_least()
{
    static_assert(logType < synapse::LogManager::LogType::LOG_MAX, "logType is too large");
    return hl_logger::logLevelAtLeast(logType, LEVEL);
}

inline bool log_level_at_least(const synapse::LogManager::LogType logType, int level)
{
    return synapse::LogManager::get_log_level(logType) <= level;
}

#define LOG_LEVEL_AT_LEAST_TRACE(log_type)     HLLOG_LEVEL_AT_LEAST_TRACE(log_type)
#define LOG_LEVEL_AT_LEAST_DEBUG(log_type)     HLLOG_LEVEL_AT_LEAST_DEBUG(log_type)
#define LOG_LEVEL_AT_LEAST_INFO(log_type)      HLLOG_LEVEL_AT_LEAST_INFO(log_type)
#define LOG_LEVEL_AT_LEAST_WARN(log_type)      HLLOG_LEVEL_AT_LEAST_WARN(log_type)
#define LOG_LEVEL_AT_LEAST_ERR(log_type)       HLLOG_LEVEL_AT_LEAST_ERR(log_type)
#define LOG_LEVEL_AT_LEAST_CRITICAL(log_type)  HLLOG_LEVEL_AT_LEAST_CRITICAL(log_type)

#define SYN_LOG(log_type, loglevel, msg, ...)      HLLOG_TYPED_PREFIXED(log_type, loglevel, msg, ## __VA_ARGS__)
#define SYN_LOG_TYPE(log_type, loglevel, msg, ...) HLLOG_TYPED(log_type, loglevel, msg, ## __VA_ARGS__)

#define SEPARATOR_STR "------------------------------------------------------------"

#define TITLE_STR(msg, ...)                                                                                            \
    "{}", fmt::format("{:=^120}", fmt::format((strlen(msg) == 0) ? "" : std::string(" ") + msg + " ", ##__VA_ARGS__))

#define LOG_TRACE(log_type, msg, ...)    HLLOG_TRACE(log_type, msg, ## __VA_ARGS__)
#define LOG_DEBUG(log_type, msg, ...)    HLLOG_DEBUG(log_type, msg, ## __VA_ARGS__)
#define LOG_INFO(log_type, msg, ...)     HLLOG_INFO(log_type, msg, ## __VA_ARGS__)
#define LOG_WARN(log_type, msg, ...)     HLLOG_WARN(log_type, msg, ## __VA_ARGS__)
#define LOG_ERR(log_type, msg, ...)      HLLOG_ERR(log_type, msg, ## __VA_ARGS__)
#define LOG_CRITICAL(log_type, msg, ...) HLLOG_CRITICAL(log_type, msg, ## __VA_ARGS__)

#define LOG_TRACE_AND_PERF(log_type, msg, ...)                                                                         \
    {                                                                                                                  \
        HLLOG_TRACE(log_type, msg, ##__VA_ARGS__);                                                                     \
        HLLOG_WARN(GC_PERF, "[LOW][DuringCompilation] " msg, ##__VA_ARGS__);                                           \
    }
#define LOG_DEBUG_AND_PERF(log_type, msg, ...)                                                                         \
    {                                                                                                                  \
        HLLOG_DEBUG(log_type, msg, ##__VA_ARGS__);                                                                     \
        HLLOG_WARN(GC_PERF, "[LOW][DuringCompilation] " msg, ##__VA_ARGS__);                                           \
    }
#define LOG_INFO_AND_PERF(log_type, msg, ...)                                                                          \
    {                                                                                                                  \
        HLLOG_INFO(log_type, msg, ##__VA_ARGS__);                                                                      \
        HLLOG_WARN(GC_PERF, "[LOW][DuringCompilation] " msg, ##__VA_ARGS__);                                           \
    }
#define LOG_WARN_AND_PERF(log_type, msg, ...)                                                                          \
    {                                                                                                                  \
        HLLOG_WARN(log_type, msg, ##__VA_ARGS__);                                                                      \
        HLLOG_WARN(GC_PERF, "[LOW][DuringCompilation] " msg, ##__VA_ARGS__);                                           \
    }
#define LOG_ERR_AND_PERF(log_type, msg, ...)                                                                           \
    {                                                                                                                  \
        HLLOG_ERR(log_type, msg, ##__VA_ARGS__);                                                                       \
        HLLOG_WARN(GC_PERF, "[LOW][DuringCompilation] " msg, ##__VA_ARGS__);                                           \
    }
#define LOG_CRITICAL_AND_PERF(log_type, msg, ...)                                                                      \
    {                                                                                                                  \
        HLLOG_CRITICAL(log_type, msg, ##__VA_ARGS__);                                                                  \
        HLLOG_WARN(GC_PERF, "[LOW][DuringCompilation] " msg, ##__VA_ARGS__);                                           \
    }

#define LOG_TRACE_T    LOG_TRACE
#define LOG_DEBUG_T    LOG_DEBUG
#define LOG_INFO_T     LOG_INFO
#define LOG_WARN_T     LOG_WARN
#define LOG_ERR_T      LOG_ERR
#define LOG_CRITICAL_T LOG_CRITICAL

#define LOG_SYN_API(msg, ...)        HLLOG_DEBUG_F(SYN_API, "SYN_API_CALL " msg, ##__VA_ARGS__);
#define LOG_SINGLETON_API(msg, ...)  HLLOG_TRACE_F(SYN_API, "SINGLETON_CALL " msg, ##__VA_ARGS__);

#define LOG_DSD_TRACE(   msg, ...)  HLLOG_TRACE_F(SYN_API, "DSD: " msg, ## __VA_ARGS__);
#define LOG_DSD_DEBUG(   msg, ...)  HLLOG_DEBUG_F(SYN_API, "DSD: " msg, ## __VA_ARGS__);
#define LOG_DSD_INFO(    msg, ...)  HLLOG_INFO_F(SYN_API,  "DSD: " msg, ## __VA_ARGS__);
#define LOG_DSD_WARN(    msg, ...)  HLLOG_WARN_F(SYN_API,  "DSD: " msg, ## __VA_ARGS__);
#define LOG_DSD_ERR(     msg, ...)  HLLOG_ERR_F(SYN_API,   "DSD: " msg, ## __VA_ARGS__);
#define LOG_DSD_CRITICAL(msg, ...)  HLLOG_CRITICAL_F(SYN_API, "DSD: " msg, ## __VA_ARGS__);

#define LOG_PERIODIC_BY_LEVEL(log_type, logLevel, period, maxNumLogsPerPeriod, msgFmt, ...)                            \
    do                                                                                                                 \
    {                                                                                                                  \
        static_assert(std::is_convertible_v<decltype(period), std::chrono::microseconds>,                              \
                      "period must be of std::chrono::duration type");                                                 \
        if (HLLOG_UNLIKELY(hl_logger::logLevelAtLeast(HLLOG_ENUM_TYPE_NAME::log_type, logLevel)))                      \
        {                                                                                                              \
            using time_point                             = std::chrono::time_point<std::chrono::steady_clock>;         \
            static time_point            epochStartPoint = std::chrono::steady_clock::now();                           \
            static std::atomic<uint64_t> msgCnt          = 0;                                                          \
            time_point                   curTimePoint    = std::chrono::steady_clock::now();                           \
            auto elapseTime = std::chrono::duration_cast<std::chrono::microseconds>(curTimePoint - epochStartPoint);   \
            if (msgCnt >= maxNumLogsPerPeriod)                                                                         \
            {                                                                                                          \
                if (elapseTime >= period)                                                                              \
                {                                                                                                      \
                    if (msgCnt > maxNumLogsPerPeriod)                                                                  \
                    {                                                                                                  \
                        HLLOG_TYPED(log_type,                                                                          \
                                    logLevel,                                                                          \
                                    msgFmt " : unpause. missed {} messages.",                                          \
                                    ##__VA_ARGS__,                                                                     \
                                    msgCnt - maxNumLogsPerPeriod - 1);                                                 \
                    }                                                                                                  \
                    else                                                                                               \
                    {                                                                                                  \
                        HLLOG_TYPED(log_type, logLevel, msgFmt, ##__VA_ARGS__);                                        \
                    }                                                                                                  \
                    msgCnt          = 0;                                                                               \
                    epochStartPoint = curTimePoint;                                                                    \
                }                                                                                                      \
                else                                                                                                   \
                {                                                                                                      \
                    if (msgCnt == maxNumLogsPerPeriod)                                                                 \
                    {                                                                                                  \
                        auto waitTime =                                                                                \
                            std::chrono::duration_cast<std::chrono::microseconds>(period - elapseTime).count();        \
                        HLLOG_TYPED(log_type,                                                                          \
                                    logLevel,                                                                          \
                                    msgFmt " : The message is generated too often. pause for {}ms.",                   \
                                    ##__VA_ARGS__,                                                                     \
                                    (waitTime + 500) / 1000);                                                          \
                    }                                                                                                  \
                    msgCnt++;                                                                                          \
                }                                                                                                      \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                HLLOG_TYPED(log_type, logLevel, msgFmt, ##__VA_ARGS__);                                                \
                if (elapseTime >= period)                                                                              \
                {                                                                                                      \
                    epochStartPoint = curTimePoint;                                                                    \
                    msgCnt          = 0;                                                                               \
                }                                                                                                      \
                else                                                                                                   \
                {                                                                                                      \
                    msgCnt++;                                                                                          \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (false)

/**
 * LOG_PERIODIC
 * @brief log frequent message - maximum maxNumLogsPerPeriod times per period
 *        after maxNumLogsPerPeriod pauses printing until the end of period
 * @param log_type logger type
 * @param period   period std::chrono::duration (e.g.milliseconds)
 * @param maxNumLogsPerPeriod max number of messages per period
 * @param msgFmt   message format
 */
#define LOG_PERIODIC_TRACE(log_type, period, maxNumLogsPerPeriod, msgFmt, ...)                                         \
    LOG_PERIODIC_BY_LEVEL(log_type, HLLOG_LEVEL_TRACE, period, maxNumLogsPerPeriod, msgFmt, ##__VA_ARGS__);
#define LOG_PERIODIC_DEBUG(log_type, period, maxNumLogsPerPeriod, msgFmt, ...)                                         \
    LOG_PERIODIC_BY_LEVEL(log_type, HLLOG_LEVEL_DEBUG, period, maxNumLogsPerPeriod, msgFmt, ##__VA_ARGS__);
#define LOG_PERIODIC_INFO(log_type, period, maxNumLogsPerPeriod, msgFmt, ...)                                          \
    LOG_PERIODIC_BY_LEVEL(log_type, HLLOG_LEVEL_INFO, period, maxNumLogsPerPeriod, msgFmt, ##__VA_ARGS__);
#define LOG_PERIODIC_WARN(log_type, period, maxNumLogsPerPeriod, msgFmt, ...)                                          \
    LOG_PERIODIC_BY_LEVEL(log_type, HLLOG_LEVEL_WARN, period, maxNumLogsPerPeriod, msgFmt, ##__VA_ARGS__);
#define LOG_PERIODIC_ERR(log_type, period, maxNumLogsPerPeriod, msgFmt, ...)                                           \
    LOG_PERIODIC_BY_LEVEL(log_type, HLLOG_LEVEL_ERROR, period, maxNumLogsPerPeriod, msgFmt, ##__VA_ARGS__);

#define LOG_STG_TRACE(   msg, ...)  HLLOG_TRACE_F(SYN_API, "STG: " msg, ## __VA_ARGS__);
#define LOG_STG_DEBUG(   msg, ...)  HLLOG_DEBUG_F(SYN_API, "STG: " msg, ## __VA_ARGS__);
#define LOG_STG_INFO(    msg, ...)  HLLOG_INFO_F(SYN_API,  "STG: " msg, ## __VA_ARGS__);
#define LOG_STG_WARN(    msg, ...)  HLLOG_WARN_F(SYN_API,  "STG: " msg, ## __VA_ARGS__);
#define LOG_STG_ERR(     msg, ...)  HLLOG_ERR_F(SYN_API,   "STG: " msg, ## __VA_ARGS__);
#define LOG_STG_CRITICAL(msg, ...)  HLLOG_CRITICAL_F(SYN_API, "STG: " msg, ## __VA_ARGS__);

#define STATIC_LOG_TRACE    LOG_TRACE
#define STATIC_LOG_DEBUG    LOG_DEBUG
#define STATIC_LOG_INFO     LOG_INFO
#define STATIC_LOG_WARN     LOG_WARN
#define STATIC_LOG_ERR      LOG_ERR
#define STATIC_LOG_CRITICAL LOG_CRITICAL


#define SET_LOGGER_SINK(log_type, pathname, lvl, size, amount)        synapse::LogManager::instance().set_logger_sink(synapse::LogManager::LogType::log_type, pathname, lvl, size, amount);
#define CREATE_LOGGER(log_type, fileName, logFileSize, logFileAmount) synapse::LogManager::instance().create_logger(synapse::LogManager::LogType::log_type, fileName, logFileSize, logFileAmount);
#define DROP_LOGGER(log_type)                                         synapse::LogManager::instance().drop_log(synapse::LogManager::LogType::log_type);

#define LOG_FUNC_SCOPE() synapse::FuncScopeLog log(__FUNCTION__)

#define TO64(x)  ((uint64_t)x)
#define TO64P(x) ((void*)x)

#define TURN_ON_TRACE_MODE_LOGGING() hl_logger::enableTraceMode(true)
#define TURN_OFF_TRACE_MODE_LOGGING() hl_logger::enableTraceMode(false)

#define LOG_TRACE_DYNAMIC_PATCHING(...) LOG_TRACE(DYN_SHAPE, __VA_ARGS__)

// compatibility section
#define SPDLOG_LEVEL_TRACE HLLOG_LEVEL_TRACE
#define SPDLOG_LEVEL_DEBUG HLLOG_LEVEL_DEBUG
#define SPDLOG_LEVEL_INFO  HLLOG_LEVEL_INFO
#define SPDLOG_LEVEL_WARN  HLLOG_LEVEL_WARN
#define SPDLOG_LEVEL_ERROR HLLOG_LEVEL_ERROR
