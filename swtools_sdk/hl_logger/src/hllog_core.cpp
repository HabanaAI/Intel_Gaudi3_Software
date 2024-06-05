#include "hl_logger/hllog.hpp"
#include "hllog_logger.hpp"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/dist_sink.h>
#include <memory>
#include <iostream>
#include <cxxabi.h>
#include <execinfo.h>
#include <deque>
#include <fstream>
#include <vector>
#include <unistd.h>
#include "ansicolor_rotating_file_sink.h"
#include <signal.h>

// rdtsc support
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

/*
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>
*/

namespace{
#ifdef _WIN32
using ColorStdoutSink = spdlog::sinks::wincolor_stdout_sink_mt;
using ColorStderrSink = spdlog::sinks::wincolor_stderr_sink_mt;
#else
using ColorStdoutSink = spdlog::sinks::ansicolor_stdout_sink_mt;
using ColorStderrSink = spdlog::sinks::ansicolor_stderr_sink_mt;
#endif

using ContextStack = std::vector<std::string>;

std::string getTidString()
{
    return fmt::format(FMT_COMPILE("[tid:{:X}]"), syscall(__NR_gettid));
}

// fork() changes tid but the change is not reflected in thread_local variables
// we must update thread_local tid
static thread_local std::string s_threadIdStr = getTidString();

static auto     startTimePoint      = std::chrono::system_clock::now();
static uint64_t startTimePointRdtsc = __rdtsc();
static std::string currentLogsFolderPath = hl_logger::getLogsFolderPathFromEnv();
}
namespace hl_logger{
namespace internal{
inline namespace v1_0{
HLLOG_API thread_local uint64_t s_threadId = syscall(__NR_gettid);
}
}
}
namespace{
std::string getPidString()
{
    return fmt::format(FMT_COMPILE("[pid:{:X}]"), getpid());
}

std::string s_processId = getPidString();

std::atomic<bool>                    s_enablePeriodicFlush {false};
std::mutex                           s_waitMtx;
std::condition_variable              s_waitCV;
std::mutex                           s_flushThreadMtx;
std::shared_ptr<std::thread>         s_flushThread;

thread_local bool s_traceMode = false;

thread_local std::string  s_threadGlobalContext;
thread_local ContextStack s_threadGlobalContextStack;

thread_local std::string  s_threadSpecialContext;
thread_local ContextStack s_threadSpecialContextStack;

void removeCurThreadContext(std::string& context, ContextStack& contextStack)
{
    if (contextStack.size())
    {
        context = contextStack.back();
        contextStack.pop_back();
    }
    else
    {
        context.clear();
    }
}
void addCurThreadContext(std::string_view newThreadContext, std::string& context, ContextStack& contextStack)
{
    if (context.size())
    {
        contextStack.push_back(s_threadGlobalContext);
    }
    context = "[C:";
    context += newThreadContext;
    context += "]";
}

int getIntEnvVar(std::string const& envVarName, int defaultValue)
{
    int         value     = defaultValue;
    const char* envVarStr = getenv(envVarName.c_str());
    if (envVarStr != nullptr)
    {
        try
        {
            value = std::stoi(envVarStr);
        }
        catch (...)
        {
            std::cerr << envVarName << " (" << envVarStr << ") must be a number. ignore\n";
        }
    }
    return value;
}

int getFlushLevel(int defaultFlushLevel)
{
    int flushLevel = getIntEnvVar("LOG_FLUSH_LEVEL", defaultFlushLevel);
    if (flushLevel < 0 || flushLevel >= HLLOG_LEVEL_OFF)
    {
        flushLevel = HLLOG_LEVEL_WARN;
    }

    return flushLevel;
}

bool getBoolEnvVar(const char* varName, bool defaultValue)
{
    const char* value = getenv(varName);
    if (value == nullptr)
    {
        return defaultValue;
    }
    std::string strValue = value;
    for(char & c : strValue)
    {
        c = tolower(c);
    }

    if (strValue == "1" || strValue == "true")
    {
        return true;
    }
    if (strValue == "0" || strValue == "false")
    {
        return false;
    }
    return defaultValue;
}

template<class THandler>
struct HandlerWithId
{
    HandlerWithId(uint64_t id, THandler handler) : id(id), handler(std::move(handler)) {}
    uint64_t id = 0;
    THandler handler;
};

const bool s_sepLogMode  = getBoolEnvVar("ENABLE_SEP_LOG_PER_THREAD", false);

template <class THandler>
struct HandlersHolder
{
    using THandlerWithId = HandlerWithId<THandler>;
    std::mutex                  handlersMutex;
    std::vector<THandlerWithId> handlers;
    unsigned                    id = 0;

    hl_logger::ResourceGuard registerHandler(THandler flushHandler)
    {
        std::lock_guard guard(handlersMutex);
        auto            curId = id++;
        handlers.emplace_back(curId, std::move(flushHandler));
        return hl_logger::ResourceGuard([curId, this]() {
            std::lock_guard guard(handlersMutex);
            auto it = std::find_if(handlers.begin(), handlers.end(), [curId](THandlerWithId const& item) {
                return item.id == curId;
            });
            if (it != handlers.end())
            {
                handlers.erase(it);
            }
        });
    }

    template <class ... TArgs>
    void invokeAllHandlers(TArgs && ... args)
    {
        std::lock_guard guard(handlersMutex);
        for (auto &handlerWithId: handlers) {
            try
            {
                handlerWithId.handler(std::forward<TArgs>(args)...);
            }
            catch(std::exception const & e)
            {
                std::cerr << "handler invocation. exception " << e.what() << " in signal handler\n";
            }
            catch(...)
            {
                std::cerr << "handler invocation. unknown exception in signal handler\n";
            }
        }
    }

    template <class TProcessFunc, class ... TArgs>
    void invokeAllHandlersAndProcess(TProcessFunc processFunc, TArgs && ... args)
    {
        std::lock_guard guard(handlersMutex);
        for (auto &handlerWithId: handlers) {
            try
            {
                processFunc(handlerWithId.handler(std::forward<TArgs>(args)...));
            }
            catch(std::exception const & e)
            {
                std::cerr << "handler invocation. exception " << e.what() << " in signal handler\n";
            }
            catch(...)
            {
                std::cerr << "handler invocation. unknown exception in signal handler\n";
            }
        }
    }
};
HandlersHolder<hl_logger::SignalHandlerV2>           s_crashHandlers;
HandlersHolder<hl_logger::FlushHandler>              s_flushHandlers;
HandlersHolder<hl_logger::internal::LazyLogsHandler> s_lazyLogsHandlers;

using SigActions            = std::array<struct sigaction, _NSIG>;
SigActions s_prevSigActions = {};
using TerminateHandler      = void();
TerminateHandler* s_prevTerminateHandler {};

void invokeCrashHandlers(int signum, bool isSevere)
{
    s_crashHandlers.invokeAllHandlers(signum, strsignal(signum), isSevere);
}
}

// make the function public to show it in the stacktrace
HLLOG_API void signalHandler(int signum, siginfo_t* info, void* data)
{
    bool isSevere = signum != SIGTERM && signum != SIGINT;
    invokeCrashHandlers(signum, isSevere);
    spdlog::details::registry::instance().flush_all();

    auto prevHandler = s_prevSigActions[signum].sa_handler;
    if (prevHandler == SIG_DFL)
    {
        // set signal handler to default
        sigaction(signum, &s_prevSigActions[signum], nullptr);
        // re-raise signal in order to invoke the previous/default  handler
        raise(signum);
    }
    else if (prevHandler == SIG_IGN)
    {
        return;
    }
    else
    {
        // call original signal handler after RT handler finished
        if (s_prevSigActions[signum].sa_flags & SA_SIGINFO)
        {
            s_prevSigActions[signum].sa_sigaction(signum, info, data);
        }
        else
        {
            prevHandler(signum);
        }
    }
}

namespace{
static void terminateHandler()
{
    invokeCrashHandlers(SIGTERM, true);
    spdlog::details::registry::instance().flush_all();

    if (s_prevTerminateHandler)
    {
        s_prevTerminateHandler();
    }
}

void installSignalsHandlers()
{
    static constexpr int signals[] = {SIGFPE,
                                      SIGILL,
                                      SIGINT,
                                      SIGSEGV,
                                      SIGTERM,
                                      SIGBUS,
                                      SIGABRT,
#ifndef WIN32
                                      SIGHUP,
                                      SIGQUIT,
                                      SIGPIPE
#endif
    };
    static constexpr bool signalsVerified = []() constexpr
    {
        for (const auto s : signals)
        {
            if (unsigned(s) >= s_prevSigActions.size())
            {
                return false;
            }
        }
        return true;
    }();

    static_assert(signalsVerified == true, "a signal number exceeds m_prevSignalHandlers size");
    // in u22 SIGSTKSZ is not a constant anymore
    // in u20 it was 8192 - keep it as a number for now
    static char           handlersStack[16 * 8192];
    static std::once_flag flag;
    std::call_once(flag, []() {
        // allocate stack for signal handlers
        // in case of stack overflow we need a separate stack for signal handling
        stack_t stackDesc;

        stackDesc.ss_sp    = handlersStack;
        stackDesc.ss_size  = sizeof(handlersStack);
        stackDesc.ss_flags = 0;
        sigaltstack(&stackDesc, nullptr);

        // register signal handles with a separate stack
        for (int s : signals)
        {
            struct sigaction action = {};
            action.sa_flags         = SA_SIGINFO | SA_ONSTACK;
            action.sa_sigaction     = &signalHandler;
            sigaction(s, &action, &s_prevSigActions[s]);
        }
        s_prevTerminateHandler = std::set_terminate(terminateHandler);
        static bool wasPeriodicFlushEnabled = false;
        pthread_atfork([]() {  // before fork
                            wasPeriodicFlushEnabled = s_enablePeriodicFlush;
                            hl_logger::enablePeriodicFlush(false);
                            hl_logger::flush();
                       },
                       []() { // parent
                            hl_logger::enablePeriodicFlush(wasPeriodicFlushEnabled);
                       },
                       [](){ // child
                            s_processId           = getPidString();
                            s_threadIdStr         = getTidString();
                            hl_logger::internal::s_threadId = syscall(__NR_gettid);
                            hl_logger::enablePeriodicFlush(wasPeriodicFlushEnabled);
                       }
                );
    });
}
struct SignalsInstaller
{
    SignalsInstaller() { installSignalsHandlers(); }
};
SignalsInstaller signalsInstaller;

void emptyLoggerError(std::string_view msg, const char  * funcName)
{
    std::cerr << "hl_logger::" << funcName
              << ": error: logger is empty. " << msg << std::endl;
    hl_logger::logStackTrace(std::cerr);
}

std::string addRankPattern(std::string const & pattern)
{
    const char* hlsId =  getenv("HLS_ID");

    const char* rankId = getenv("ID");
    if (rankId == nullptr)
    {
        rankId = getenv("OMPI_COMM_WORLD_RANK");
    }
    std::string hlsIdPattern = hlsId ? (std::string("[hls:") + hlsId + "]") : std::string();
    std::string rankPattern  = rankId ? (std::string("[rank:") + rankId + "]") : std::string();

    bool rankAlreadyInPattern = (!hlsIdPattern.empty() && pattern.find(hlsIdPattern) != std::string::npos) ||
                                (!rankPattern.empty() && pattern.find(rankPattern) != std::string::npos);
    if (rankAlreadyInPattern)
    {
        return pattern;
    }
    auto userMessagePos = pattern.find("%v");
    return pattern.substr(0, userMessagePos) + hlsIdPattern + rankPattern + (userMessagePos != std::string::npos ? "%v" : "");
}
}  // anonymous namespace

#define VERIFY_LOGGER(msg)                                                         \
if (HLLOG_UNLIKELY(logger == nullptr))                                             \
{                                                                                  \
    emptyLoggerError(msg, __FUNCTION__);                                           \
    return;                                                                        \
}

#define VERIFY_LOGGER_R(msg, ret)                                                  \
if (HLLOG_UNLIKELY(logger == nullptr))                                             \
{                                                                                  \
    emptyLoggerError(msg, __FUNCTION__);                                           \
    return ret;                                                                    \
}

extern const char* SWTOOLS_SDK_SHA1_VERSION;

namespace hl_logger{
static std::string getDefaultLogPattern(LoggerCreateParams const& params)
{
    std::string pattern;
    if (params.printTime)
    {
        bool printDate = getBoolEnvVar("PRINT_DATE", false);
        bool printTime = getBoolEnvVar("PRINT_TIME", true);
        if (printDate && printTime)
        {
            pattern += "[%Y-%m-%d %T.%f]";
        }
        else if (printDate)
        {
            pattern += "[%Y-%m-%d]";
        }
        else if (printTime)
        {
            pattern += "[%T.%f]";
        }
    }
    if (params.printLoggerName)
    {
        if (params.loggerNameLength)
        {
            pattern += "[%-" + std::to_string(params.loggerNameLength) + "n]";
        }
        else
        {
            pattern += "[%n]";
        }
    }
    if (params.logLevelStyle != LoggerCreateParams::LogLevelStyle::off)
    {
        if (params.logLevelStyle == LoggerCreateParams::LogLevelStyle::full_name)
        {
            pattern += "[%^%-5l%$]";
        }
        else
        {
            pattern += "[%L]";
        }
    }
    if (getBoolEnvVar("PRINT_RANK", params.printRank))
    {
        pattern = addRankPattern(pattern);
    }

    pattern += "%v";
    return pattern;
}


static std::string getUpdatedFileName(std::string filename, const std::string& logsRootDir)
{
    assert(!filename.empty());

    if (filename.find("/") != 0)
    {
        if (!logsRootDir.empty())
        {
            filename = logsRootDir + "/" + filename;
        }
    }

    return filename;
}

static std::string getUpdatedFileName(const std::string& filename)
{
    std::string directory = getLogsFolderPath();
    return getUpdatedFileName(filename, directory);
}

struct SinkCacheEntry {
    spdlog::sink_ptr proxySink; // instance of dist_sink that points to the dst sink (file sink)
    std::function<spdlog::sink_ptr(const std::string&)> sinkCreatorFn; // factory function creating dst sink (file sink)
};

using SinksCache = std::unordered_map<std::string, SinkCacheEntry>;
using SinksDB    = std::vector<spdlog::sink_ptr>;

static SinksCache sinksCache;
static std::recursive_mutex sinkCacheMtx;

static void addNewSink(std::string const&                           logFilename,
                       std::function<spdlog::sink_ptr(std::string)> sinkCreator,
                       SinksDB&                                     sinks)
{
    auto it = sinksCache.find(logFilename);
    if (it == sinksCache.end())
    {
        auto proxy_sink = std::make_shared<spdlog::sinks::dist_sink_mt>();
        proxy_sink->add_sink(sinkCreator(getLogsFolderPath()));

        it = sinksCache.insert({logFilename, {proxy_sink, sinkCreator}}).first;
    }
    sinks.push_back(it->second.proxySink);
}

template <class TBaseSink>
class SinkWithBuffer : public spdlog::sinks::sink
{
public:
    template<class ... Args>
    SinkWithBuffer(uint64_t fileBufferSize, Args const & ... args)
    : _fileBuffer(fileBufferSize ? std::make_unique<char[]>(fileBufferSize) : nullptr)
    , _baseSink(args..., [this, fileBufferSize](){
        spdlog::file_event_handlers fileHandler;
        if (fileBufferSize != 0)
        {
            fileHandler.after_open = [bufferPtr = _fileBuffer.get(), fileBufferSize](const spdlog::filename_t& filename,
                                                                                     std::FILE*          file_stream) {
                setvbuf(file_stream, bufferPtr, _IOFBF, fileBufferSize);
            };
        }
        return fileHandler;
    }())
    {
    }
    void log(const spdlog::details::log_msg &msg) override
    {
        _baseSink.log(msg);
    };
    void flush() override
    {
        _baseSink.flush();
    }
    void set_pattern(const std::string &pattern) override
    {
        _baseSink.set_pattern(pattern);
    }
    void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter) override
    {
        _baseSink.set_formatter(std::move(sink_formatter));
    }
    const spdlog::filename_t filename() const
    {
        return _baseSink.filename();
    }
private:
    std::unique_ptr<char[]> _fileBuffer;
    TBaseSink               _baseSink;

};
inline namespace v1_3{
LoggerSPtr createLogger(std::string_view loggerName, LoggerCreateParams const& params)
{
    std::lock_guard   guard(sinkCacheMtx);

    SinksDB sinks;

    const bool enableConsole  = getBoolEnvVar("ENABLE_CONSOLE", false) &&
                                params.consoleStream != LoggerCreateParams::ConsoleStream::disabled;
    const std::string pattern = params.spdlogPattern.empty() ? getDefaultLogPattern(params) : params.spdlogPattern;
    spdlog::sink_ptr console_sink;
    if (enableConsole)
    {
        if (params.consoleStream == LoggerCreateParams::ConsoleStream::std_out)
        {
            console_sink = std::make_shared<ColorStdoutSink>();
        }
        else
        {
            console_sink =  std::make_shared<ColorStderrSink>();
        }
        sinks.push_back(console_sink);

        auto console_pattern = getBoolEnvVar("PRINT_RANK", true) ? addRankPattern(pattern) : pattern;
        console_sink->set_pattern(console_pattern);
    }
    const bool enableColors = getBoolEnvVar("ENABLE_LOG_FILE_COLORS", false);
    int logFileSize = getIntEnvVar("LOG_FILE_SIZE", params.logFileSize);
    logFileSize     = logFileSize ? logFileSize : 1024 * 1024 * 5;
    addNewSink(
        params.logFileName,
        [params, logFileSize, enableColors, pattern](const std::string& logsRootDir)
        {
          const std::string logFullOath = getUpdatedFileName(params.logFileName, logsRootDir);
          auto sink = std::make_shared<
              SinkWithBuffer<spdlog::sinks::ansicolor_rotating_file_sink_mt>>(
              params.logFileBufferSize, logFullOath, logFileSize,
              params.logFileAmount, params.rotateLogfileOnOpen, enableColors);
          sink->set_pattern(pattern);
          return sink;
        },
        sinks);
    const bool disableSeparateLogsFiles = getBoolEnvVar("DISABLE_SEPARATE_LOG_FILES", false);
    if (!params.separateLogFile.empty() && disableSeparateLogsFiles == false)
    {
        addNewSink(
            params.separateLogFile,
            [params, pattern](const std::string& logsRootDir)
            {
              const std::string separateFilename = getUpdatedFileName(params.separateLogFile, logsRootDir);
              std::ofstream logToClean(separateFilename);
              logToClean.close();
              auto sink = std::make_shared<
                  SinkWithBuffer<spdlog::sinks::basic_file_sink_mt>>(
                  params.separateLogFileBufferSize, separateFilename, false);
              sink->set_pattern(pattern);
              return sink;
            },
            sinks);
    }
    auto combined_logger = std::make_shared<Logger>(std::string(loggerName), begin(sinks), end(sinks));
    // reserve space for more items in order to eliminate possible reallocation at runtime
    combined_logger->sinks().reserve(sinks.size() + 10);

    int printSpecialContext = (int)getBoolEnvVar("PRINT_SPECIAL_CONTEXT", params.printSpecialContext);
    int printThreadID       = (int)getBoolEnvVar("PRINT_TID", params.printThreadID);
    int printProcessID      = (int)getBoolEnvVar("PRINT_PID", params.printProcessID);
    int printFileLine       = (int)getBoolEnvVar("PRINT_FILE_AND_LINE", params.forcePrintFileLine);

    combined_logger->printOptions =
        Logger::PrintOptions(printThreadID + (printProcessID << 1) + (printSpecialContext << 2) + (printFileLine << 3));
    combined_logger->sepLogPerThread = params.sepLogPerThread;
    combined_logger->set_level(spdlog::level::level_enum::trace);
    combined_logger->spdlogPattern = pattern;
    if (params.defaultLoggingLevel != 0xFF)
    {
        const int defaultLevel = getDefaultLoggingLevel(loggerName, params.defaultLoggingLevel);
        const int loggingLevel = params.forceDefaultLoggingLevel ? params.defaultLoggingLevel : defaultLevel;
        combined_logger->set_level(spdlog::level::level_enum(loggingLevel));
    }
    if (params.defaultLazyLoggingLevel != 0xFF)
    {
        const int defaultLazyLevel = getDefaultLazyLoggingLevel(loggerName, params.defaultLazyLoggingLevel);
        const int lazyLoggingLevel = params.forceDefaultLazyLoggingLevel ? params.defaultLazyLoggingLevel : defaultLazyLevel;
        combined_logger->lazy_level = lazyLoggingLevel;
    }
    if (params.registerLogger)
    {
        spdlog::register_logger(combined_logger);
    }
    combined_logger->flush_on(spdlog::level::level_enum(getFlushLevel(params.loggerFlushLevel)));
    return combined_logger;
}

}
inline namespace v1_0{

void setLoggingLevel(LoggerSPtr const& logger, int newLevel)
{
    VERIFY_LOGGER("");
    logger->set_level(spdlog::level::level_enum(newLevel));
}

void setLazyLoggingLevel(LoggerSPtr const& logger, int newLevel)
{
    VERIFY_LOGGER("");
    logger->lazy_level = newLevel;
}

int getLoggingLevel(LoggerSPtr const& logger)
{
    return logger ? logger->level() : HLLOG_LEVEL_OFF;
}

int getLazyLoggingLevel(LoggerSPtr const& logger)
{
    return logger ? logger->lazy_level : HLLOG_LEVEL_OFF;
}

void flush(LoggerSPtr const& logger)
{
    VERIFY_LOGGER("");
    logger->flush();
}

void flush()
{
    s_flushHandlers.invokeAllHandlers();
}

static void periodicFlush()
{
    while (s_enablePeriodicFlush)
    {
        {
            std::unique_lock lock(s_waitMtx);
            s_waitCV.wait_for(lock, std::chrono::seconds(1));
        }
        if (s_enablePeriodicFlush)
        {
            flush();
        }
    }
}

void enablePeriodicFlush(bool enable /* = true*/)
{
    std::lock_guard lock(s_flushThreadMtx);
    if (s_enablePeriodicFlush == enable) return;
    if (enable)
    {
        s_enablePeriodicFlush = true;
        s_flushThread         = std::shared_ptr<std::thread>(new std::thread(&periodicFlush), [](std::thread* pThread) {
            {
                std::unique_lock lock(s_waitMtx);
                s_enablePeriodicFlush = false;
                s_waitCV.notify_all();
            }
            if (pThread) pThread->join();
            delete pThread;
        });
    }
    else
    {
        s_flushThread.reset();
    }
}

void addFileSink(LoggerSPtr const& logger,
                 std::string_view  logFileName,
                 size_t            logFileSize,
                 size_t            logFileAmount,
                 int               loggingLevel)
{
    VERIFY_LOGGER(logFileName);

    auto curLoggingLevel = logger->level();
    logger->set_level(spdlog::level::level_enum::off);
    const bool enableColors = getBoolEnvVar("ENABLE_LOG_FILE_COLORS", false);
    const std::string logFileNameStr(logFileName);
    addNewSink(
        logFileNameStr,
        [logFileNameStr, logFileSize, logFileAmount, enableColors](const std::string& logsRootPath) {
          std::string file_path = getUpdatedFileName(logFileNameStr, logsRootPath);
          return std::make_shared<
              spdlog::sinks::ansicolor_rotating_file_sink_mt>(
              file_path, logFileSize, logFileAmount, false, enableColors);
        },
        logger->sinks());

    auto sinkLoggingLevel = loggingLevel != defaultLoggingLevel ? spdlog::level::level_enum(loggingLevel) : curLoggingLevel;
    logger->sinks().back()->set_level(sinkLoggingLevel);
    logger->sinks().back()->set_pattern(logger->spdlogPattern);
    logger->set_level(curLoggingLevel);
}

static void logFilePerThread(LoggerSPtr const& logger, int logLevel, std::string_view preformattedString)
{
    static std::atomic<unsigned>         threadIdx {0};
    thread_local static spdlog::sink_ptr sinkPerThread = []() {
        // create new sink and assign s_multiGraphLogSink to it
        unsigned          newThreadIdx = threadIdx++;
        const std::string filename     = getUpdatedFileName(fmt::format("per_thread_{}.log", newThreadIdx));
        std::ofstream     logToClean(filename);
        logToClean.close();

        spdlog::sink_ptr sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, false);
        return sink;
    }();

    // add sink to logger sinks vector, and write to log
    logger->sinks().push_back(sinkPerThread);
    logger->log(spdlog::level::level_enum(logLevel), preformattedString);
    logger->sinks().pop_back();
}

#define MSG_FMT " {}"
#define FILE_LN_FMT "[{}::{}]"

void log(hl_logger::LoggerSPtr const& logger,
         int logLevel,
         std::string_view msg,
         std::string_view file,
         int line,
         bool forcePrintFileLine)
try
{
    VERIFY_LOGGER(msg);

    if (HLLOG_UNLIKELY(s_traceMode))
    {
        logLevel = HLLOG_LEVEL_TRACE;
    }

    fmt::memory_buffer buf;
    Logger::PrintOptions printOptions(Logger::PrintOptions((int(forcePrintFileLine) << 3) | int(logger->printOptions)));
    switch (printOptions)
    {
        case Logger::PrintOptions::pid_tid_specialContext_file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{}{}[{}::{}] {}"),
                           s_processId,
                           s_threadIdStr,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::tid_specialContext_file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{}[{}::{}] {}"),
                           s_threadIdStr,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::pid_specialContext_file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{}[{}::{}] {}"),
                           s_processId,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::specialContext_file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}[{}::{}] {}"),
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::pid_tid_file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{}[{}::{}] {}"),
                           s_processId,
                           s_threadIdStr,
                           s_threadGlobalContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::tid_file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}[{}::{}] {}"),
                           s_threadIdStr,
                           s_threadGlobalContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::pid_file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}[{}::{}] {}"),
                           s_processId,
                           s_threadGlobalContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::file:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}[{}::{}] {}"),
                           s_threadGlobalContext,
                           file, line,
                           msg);
            break;
        case Logger::PrintOptions::pid_tid_specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{}{} {}"),
                           s_processId,
                           s_threadIdStr,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::tid_specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{} {}"),
                           s_threadIdStr,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::pid_specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{} {}"),
                           s_processId,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{} {}"),
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::pid_tid:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{}{} {}"),
                           s_processId,
                           s_threadIdStr,
                           s_threadGlobalContext,
                           msg);
            break;
        case Logger::PrintOptions::tid:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{} {}"),
                           s_threadIdStr,
                           s_threadGlobalContext,
                           msg);
            break;
        case Logger::PrintOptions::pid:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE("{}{} {}"),
                           s_processId,
                           s_threadGlobalContext,
                           msg);
            break;
        case Logger::PrintOptions::none:
            fmt::format_to(std::back_inserter(buf), FMT_COMPILE("{} {}"), s_threadGlobalContext, msg);
            break;
    }
    std::string_view preformattedString{buf.data(), buf.size()};

    if (HLLOG_UNLIKELY(logger->sepLogPerThread + s_sepLogMode == 2)) {
        logFilePerThread(logger, logLevel, preformattedString);
    } else {
        logger->log(spdlog::level::level_enum(logLevel), preformattedString);
    }
}
catch(...)
{

}

static std::string_view to_string(spdlog::level::level_enum level)
{
    switch (level)
    {
        case spdlog::level::trace: return "trace";
        case spdlog::level::debug: return "debug";
        case spdlog::level::info:  return "info ";
        case spdlog::level::warn:  return "warn ";
        case spdlog::level::err:   return "error";
        case spdlog::level::critical: return "critical";
        default: return "";
    }
}

#define TID_FMT "[tid:{:X}]"
#define PREFIX_FMT "[{:%H:%M:%S}.{:>06}][{:<{}}][{}]"
#undef MSG_FMT
#define MSG_FMT " {}"
static void logLazyMsgAfterProcessing(LoggerSPtr const&   logger,
                                     std::string const & loggerName,
                                     unsigned            loggerNameLength,
                                     int                 logLevel,
                                     std::string_view    msg,
                                     internal::TimePoint timePoint,
                                     uint64_t            tid)
try
{
    VERIFY_LOGGER(msg);
    if (HLLOG_UNLIKELY(s_traceMode))
    {
        logLevel = HLLOG_LEVEL_TRACE;
    }

    fmt::memory_buffer buf;
    Logger::PrintOptions printOptions(Logger::PrintOptions(int(logger->printOptions)));
    auto uSeconds = std::chrono::duration_cast<std::chrono::microseconds>(timePoint.time_since_epoch()).count() % 1000000ull;
    auto levelStr = to_string(spdlog::level::level_enum(logLevel));
    switch (printOptions)
    {
        case Logger::PrintOptions::pid_tid_specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT "{}" TID_FMT "{}{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           s_processId,
                           tid,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::tid_specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT TID_FMT "{}{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           tid,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::pid_specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT "{}{}{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           s_processId,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::specialContext:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT "{}{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           s_threadGlobalContext,
                           s_threadSpecialContext,
                           msg);
            break;
        case Logger::PrintOptions::pid_tid:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT "{}" TID_FMT "{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           s_processId,
                           tid,
                           s_threadGlobalContext,
                           msg);
            break;
        case Logger::PrintOptions::tid:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT TID_FMT "{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           tid,
                           s_threadGlobalContext,
                           msg);
            break;
        case Logger::PrintOptions::pid:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT "{}{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           s_processId,
                           s_threadGlobalContext,
                           msg);
            break;
        case Logger::PrintOptions::none:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT "{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           s_threadGlobalContext,
                           msg);
            break;
        default:
            fmt::format_to(std::back_inserter(buf),
                           FMT_COMPILE(PREFIX_FMT TID_FMT "{}" MSG_FMT),
                           timePoint,
                           uSeconds,
                           loggerName,
                           loggerNameLength,
                           levelStr,
                           tid,
                           s_threadGlobalContext,
                           msg);
            break;
    };
    std::string_view formattedFullString{buf.data(), buf.size()};

    logger->log(spdlog::level::err, formattedFullString);
}
catch(std::exception const & e)
{
    std::cerr << "exception in hl_logger::log: " << e.what() << ". msg: " << msg << std::endl;
}
catch(...)
{
    std::cerr << "unknown exception in hl_logger::log. msg: " << msg << std::endl;
}

void logAllLazyLogs(std::string_view filename)
{
    auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(std::string(filename), false);
    auto logger = std::make_shared<Logger>("LAZY", sink);
    logAllLazyLogs(logger);
}

void logAllLazyLogs(LoggerSPtr logger)
{
    internal::LazyLogInfoVector lazyLogsInfo;
    s_lazyLogsHandlers.invokeAllHandlersAndProcess([&](internal::LazyLogInfoVector lazyLogs){
        lazyLogsInfo.insert(lazyLogsInfo.end(),
                           std::make_move_iterator(lazyLogs.begin()),
                           std::make_move_iterator(lazyLogs.end()));
    });
    if (lazyLogsInfo.empty())
    {
        return;
    }
    auto itMaxLen = std::max_element(lazyLogsInfo.begin(), lazyLogsInfo.end(), [](auto const & lhs, auto const & rhs){
        return lhs.loggerName.size() < rhs.loggerName.size();
    });
    unsigned maxLoggerNameLen = 5;
    if (itMaxLen != lazyLogsInfo.end())
    {
        maxLoggerNameLen = itMaxLen->loggerName.size();
    }

    std::vector<internal::FormattedLazyLogItemOptional>  lazyLogItems;
    for (auto & lazyLogInfo : lazyLogsInfo)
    {
        lazyLogItems.push_back(lazyLogInfo.getNextLogItemFunc());
    }
    logger->set_pattern("%v");
    while(true)
    {
        auto minTimePoint = std::chrono::system_clock::now() + std::chrono::hours(100);
        unsigned minIndex = -1;
        for (unsigned i = 0; i < lazyLogItems.size(); ++i) {
            auto & item = lazyLogItems[i];
            if (item.has_value() && item->timePoint < minTimePoint)
            {
                minIndex = i;
                minTimePoint = item->timePoint;
            }
        }
        if (minIndex == -1) break;
        auto & item = lazyLogItems[minIndex];
        logLazyMsgAfterProcessing(logger, lazyLogsInfo[minIndex].loggerName, maxLoggerNameLen, item->logLevel, item->msg, item->timePoint, item->tid);
        item = lazyLogsInfo[minIndex].getNextLogItemFunc();
    };
    logger->set_pattern(logger->spdlogPattern);
}

uint8_t getDefaultLoggingLevel(std::string_view loggerName, int defaultLevel)
try
{
    int logLevel = getIntEnvVar("LOG_LEVEL_ALL", defaultLevel);

    auto             prefixPos = loggerName.find('_');
    std::string_view prefix    = loggerName.substr(0, prefixPos);
    logLevel                   = getIntEnvVar("LOG_LEVEL_ALL_" + std::string(prefix), logLevel);
    logLevel                   = getIntEnvVar("LOG_LEVEL_" + std::string(loggerName), logLevel);

    return logLevel;
}
catch(...)
{
    return defaultLoggingLevel;
}

uint8_t getDefaultLazyLoggingLevel(std::string_view loggerName, int defaultLevel)
try
{
    int logLevel = getIntEnvVar("LAZY_LOG_LEVEL_ALL", defaultLevel);

    auto             prefixPos = loggerName.find('_');
    std::string_view prefix    = loggerName.substr(0, prefixPos);
    logLevel                   = getIntEnvVar("LAZY_LOG_LEVEL_ALL_" + std::string(prefix), logLevel);
    logLevel                   = getIntEnvVar("LAZY_LOG_LEVEL_" + std::string(loggerName), logLevel);

    // don't allow users to make lazy logs more restrictive than what a developer set
    if (logLevel > defaultLevel)
    {
        // if user enabled experimental flags - we trust the user and don't force defaultLevel as maximum
        if (getBoolEnvVar("ENABLE_EXPERIMENTAL_FLAGS", false) == false && getBoolEnvVar("EXP_FLAGS", false) == false)
        {
            logLevel = defaultLevel;
        }
    }
    return logLevel;
}
catch(...)
{
    return defaultLoggingLevel;
}

uint32_t getLazyQueueSize(std::string_view loggerName, uint32_t defaultQueueSize)
try
{
    uint32_t queueSize = getIntEnvVar("LAZY_LOG_QUEUE_SIZE_ALL", defaultQueueSize);

    auto             prefixPos = loggerName.find('_');
    std::string_view prefix    = loggerName.substr(0, prefixPos);
    queueSize                  = getIntEnvVar("LAZY_LOG_QUEUE_SIZE_ALL_" + std::string(prefix), queueSize);
    queueSize                  = getIntEnvVar("LAZY_LOG_QUEUE_SIZE_" + std::string(loggerName), queueSize);

    return queueSize;
}
catch(...)
{
    return defaultQueueSize;
}

static std::string_view userStackTraceDlim = "===============================================================================";
static std::string_view userStackTraceMsg  = "====================== USER CODE STACK TRACE START POINT ======================";

void logStackTrace(LoggerSPtr const& logger, int logLevel)
try
{
    VERIFY_LOGGER("");

    /* // boost stacktrace
    unsigned i = 0;
    for (auto const & frame : boost::stacktrace::stacktrace())
    {
        auto adr =  (uint64_t)frame.address();
        std::string s= fmt::format("{} {:X} {} at {}:{}", i++, adr, frame.name(), frame.source_file(),
    frame.source_line()); std::cout << s <<std::endl;
    }
    */
    constexpr int btSize         = 30;
    void*         buffer[btSize] = {};

    int    numPtrs = backtrace(buffer, btSize);
    char** strings = backtrace_symbols(buffer, numPtrs);
    if (strings == nullptr)
    {
        log(logger, logLevel, "no backtrace_symbols");
    }
    else
    {
        log(logger, logLevel, fmt::format("backtrace (up to {})", btSize));
        for (int j = 0; j < numPtrs; j++)
        {
            std::string s            = strings[j];
            auto        namePosStart = s.find('(') + 1;
            auto        namePosEnd   = s.find("+0x", namePosStart);
            std::string symbol       = s.substr(namePosStart, namePosEnd - namePosStart);
            int         status       = -4;

            std::unique_ptr<char, decltype(std::free)&> demangled_name {nullptr, std::free};
            demangled_name.reset(abi::__cxa_demangle(symbol.c_str(), nullptr, nullptr, &status));
            if (demangled_name != nullptr && status == 0)
            {
                s = s.substr(0, namePosStart) + demangled_name.get() + s.substr(namePosEnd, -1);
            }
            log(logger, logLevel, s);
            if (s.find("hl_logger.so(signalHandler") != std::string::npos)
            {
                log(logger, logLevel, userStackTraceDlim);
                log(logger, logLevel, userStackTraceMsg);
                log(logger, logLevel, userStackTraceDlim);
            }
        }
    }
    free(strings);  // based on backtrace_symbols documentation, only "strings" needs to be freed
}
catch(...)
{

}

HLLOG_API void logStackTrace(std::ostream & ostream)
try
{
    constexpr int btSize         = 30;
    void*         buffer[btSize] = {};

    int    numPtrs = backtrace(buffer, btSize);
    char** strings = backtrace_symbols(buffer, numPtrs);
    if (strings == nullptr)
    {
        ostream <<  "no backtrace_symbols" << std::endl;
    }
    else
    {
        ostream <<  "backtrace (up to " << btSize << ")" << std::endl;
        for (int j = 0; j < numPtrs; j++)
        {
            std::string s            = strings[j];
            auto        namePosStart = s.find('(') + 1;
            auto        namePosEnd   = s.find("+0x", namePosStart);
            std::string symbol       = s.substr(namePosStart, namePosEnd - namePosStart);
            int         status       = -4;

            std::unique_ptr<char, decltype(std::free)&> demangled_name {nullptr, std::free};
            demangled_name.reset(abi::__cxa_demangle(symbol.c_str(), nullptr, nullptr, &status));
            if (demangled_name != nullptr && status == 0)
            {
                s = s.substr(0, namePosStart) + demangled_name.get() + s.substr(namePosEnd, -1);
            }
            ostream <<  s << std::endl;
            if (s.find("hl_logger.so(signalHandler") != std::string::npos)
            {
                ostream << userStackTraceDlim << std::endl;
                ostream << userStackTraceMsg << std::endl;
                ostream << userStackTraceDlim << std::endl;
            }
        }
    }
    free(strings);  // based on backtrace_symbols documentation, only "strings" needs to be freed
}
catch(...)
{

}

static void addSuffixAndCreateFolder(std::string & logsFolderPath, const char * suffix)
{
    if (suffix != nullptr)
    {
        logsFolderPath += "/";
        logsFolderPath += suffix;
        if (mkdir(logsFolderPath.c_str(), 0777) != 0 && errno != EEXIST) assert(0);
    }
}

std::string getLogsFolderPathFromEnv()
{
    std::string logsFolderPath;
    const char* habanaLogDirEnv = getenv("HABANA_LOGS");
    if (habanaLogDirEnv != nullptr)
    {
        logsFolderPath = habanaLogDirEnv;
    }
    else
    {
        const char* homeEnv = getenv("HOME");
        if (homeEnv == nullptr)
        {
            return logsFolderPath;
        }
        logsFolderPath = std::string(homeEnv) + "/.habana_logs";
    }

    // make sure top log dir exist
    if (mkdir(logsFolderPath.c_str(), 0777) != 0 && errno != EEXIST) assert(0);

    addSuffixAndCreateFolder(logsFolderPath, getenv("HLS_ID"));

    char* rankId = getenv("ID");
    if (rankId == nullptr)
    {
        rankId = getenv("OMPI_COMM_WORLD_RANK");
    }
    addSuffixAndCreateFolder(logsFolderPath, rankId);

    std::unique_ptr<char, decltype(std::free)*> fullpath {realpath(logsFolderPath.c_str(), nullptr), std::free};
    if (fullpath != nullptr)
    {
        logsFolderPath = fullpath.get();
    }
    else
    {
        assert(0);
    }
    return logsFolderPath;
}

std::string getLogsFolderPath()
{
    std::lock_guard guard(sinkCacheMtx);
    return currentLogsFolderPath;
}

LoggerSPtr getRegisteredLogger(std::string_view loggerName)
{
    return std::dynamic_pointer_cast<Logger>(spdlog::get(std::string(loggerName)));
}

void dropRegisteredLogger(std::string_view loggerName)
{
    spdlog::drop(std::string(loggerName));
}

void dropAllRegisteredLoggers()
{
    spdlog::drop_all();
}

void refreshInternalSinkCache()
{
    std::lock_guard guard(sinkCacheMtx);
    for (auto it = sinksCache.begin(); it != sinksCache.end(); )
    {
        if (it->second.proxySink.use_count() == 1)
        {
            it = sinksCache.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void addCurThreadGlobalContext(std::string_view threadContext)
{
    addCurThreadContext(threadContext, s_threadGlobalContext, s_threadGlobalContextStack);
}

void removeCurThreadGlobalContext()
{
    removeCurThreadContext(s_threadGlobalContext, s_threadGlobalContextStack);
}

void addCurThreadSpecialContext(std::string_view threadContext)
{
    addCurThreadContext(threadContext, s_threadSpecialContext, s_threadSpecialContextStack);
}

void removeCurThreadSpecialContext()
{
    removeCurThreadContext(s_threadSpecialContext, s_threadSpecialContextStack);
}

void enableTraceMode(bool enableTraceMode)
{
    s_traceMode = enableTraceMode;
}

SinksSPtr getSinks(LoggerSPtr const& logger)
{
    VERIFY_LOGGER_R("", nullptr);

    auto sinks = std::make_shared<Sinks>();
    sinks->sinks = logger->sinks();
    return sinks;
}

template <class TFileSink>
static std::string getSinkFilename(spdlog::sink_ptr sink)
{
    auto sink_wrapper = std::dynamic_pointer_cast<spdlog::sinks::dist_sink_mt>(sink);
    if (sink_wrapper == nullptr)
    {
        return "";
    }

    if (auto fileSink = std::dynamic_pointer_cast<TFileSink>(sink_wrapper->sinks()[0]))
    {
        return fileSink->filename();
    }
    if (auto fileSink = std::dynamic_pointer_cast<SinkWithBuffer<TFileSink>>(sink_wrapper->sinks()[0]))
    {
        return fileSink->filename();
    }
    return "";
}

static std::string getSinkFilename(spdlog::sink_ptr sink)
{
    auto filename = getSinkFilename<spdlog::sinks::basic_file_sink_mt>(sink);
    if (filename.empty())
    {
        filename = getSinkFilename<spdlog::sinks::ansicolor_rotating_file_sink_mt>(sink);
    }

    return filename;
}

std::vector<std::string> getSinksFilenames(LoggerSPtr const& logger)
{
    std::vector<std::string> filenames;
    VERIFY_LOGGER_R("", filenames);
    for (auto &sink: logger->sinks())
    {
        auto filename = getSinkFilename(sink);
        if (!filename.empty())
        {
            filenames.push_back(filename);
        }
    }
    return filenames;
}

SinksSPtr setSinks(LoggerSPtr const& logger, SinksSPtr newSinks)
{
    VERIFY_LOGGER_R("", nullptr);

    auto oldSinks = std::make_shared<Sinks>();
    oldSinks->sinks = std::move(logger->sinks());
    logger->sinks().clear();
    if (newSinks != nullptr)
    {
        logger->sinks() = newSinks->sinks;
    }
    return oldSinks;
}

static bool removeConsole(LoggerSPtr const& logger)
{
    for (auto it = logger->sinks().begin(); it != logger->sinks().end(); ++it)
    {
        if (std::dynamic_pointer_cast<ColorStdoutSink>(*it) != nullptr)
        {
            logger->sinks().erase(it);
            return true;
        }
    }
    return false;
}

void setLogsFolderPath(const std::string& logsRootDir)
{
    std::lock_guard guard(sinkCacheMtx);
    for (auto &[fileName, sinkCacheEntry] : sinksCache)
    {
        std::string newSinkPath = logsRootDir + "/" + fileName;
        std::string currSinkPath = getSinkFilename(sinkCacheEntry.proxySink);

        if (newSinkPath != currSinkPath)
        {
            auto proxySink = std::dynamic_pointer_cast<spdlog::sinks::dist_sink_mt>(sinkCacheEntry.proxySink);
            proxySink->set_sinks({sinkCacheEntry.sinkCreatorFn(logsRootDir)});
        }
    }

    currentLogsFolderPath = logsRootDir;
}

void setLogsFolderPathFromEnv()
{
    setLogsFolderPath(getLogsFolderPathFromEnv());
}

ResourceGuard addConsole(LoggerSPtr const& logger)
{
    VERIFY_LOGGER_R("", {});

    for (auto& sink : logger->sinks())
    {
        if (std::dynamic_pointer_cast<ColorStdoutSink>(sink) != nullptr)
        {
            return {};
        }
    }
    logger->sinks().push_back(std::make_shared<ColorStdoutSink>());
    return ResourceGuard([logger](){
        removeConsole(logger);
    });
}

VersionInfo getVersion()
{
    VersionInfo versionInfo;
    versionInfo.commitSHA1 = SWTOOLS_SDK_SHA1_VERSION;
    return versionInfo;
}

ResourceGuard registerSignalHandler(SignalHandler crashSignalHandler)
{
    return registerSignalHandler([crashSignalHandler = std::move(crashSignalHandler)](int signal, const char* signalStr, bool){
        crashSignalHandler(signal, signalStr);
    });
}

ResourceGuard registerSignalHandler(SignalHandlerV2 crashSignalHandler)
{
    return s_crashHandlers.registerHandler(std::move(crashSignalHandler));
}

ResourceGuard registerFlushHandler(FlushHandler flushHandler)
{
    return s_flushHandlers.registerHandler(std::move(flushHandler));
}
} // vXX
namespace internal{
inline namespace v1_0{
// Calculate the CPU frequency
static const uint64_t freqMHz = [](){
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;

    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu MHz") == 0) {
            size_t colonPos = line.find(':');
            if (colonPos != std::string::npos) {
                std::string cpuMHz = line.substr(colonPos + 1);
                // Remove leading/trailing spaces
                cpuMHz.erase(0, cpuMHz.find_first_not_of(" \t"));
                cpuMHz.erase(cpuMHz.find_last_not_of(" \t") + 1);
                double freqMHz = atof(cpuMHz.c_str());
                return uint64_t(freqMHz + 0.5);
            }
        }
    }

    // cpu frequency not found - try to evaluate
    auto start = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto end = __rdtsc();
    uint64_t freqMHz = (end - start) / 1'000'000;
    return freqMHz;
}();

static inline uint64_t rdtscToNs(uint64_t tsc) {
    // Convert TSC value to nanoseconds
    uint64_t ns = (tsc * 1'000) / freqMHz;
    return ns;
}

TimePoint tscToRt(uint64_t tsc)
{
    return startTimePoint + std::chrono::nanoseconds(rdtscToNs(tsc - startTimePointRdtsc));
}
ResourceGuard registerLazyLogsHandler(LazyLogsHandler lazyLogsHandler)
{
    return s_lazyLogsHandlers.registerHandler(std::move(lazyLogsHandler));
}

void logLazy(LoggerSPtr const& logger, int logLevel, CreateFormatterFunc * createFormatterFunc, void * argsAsTupleVoidPtr)
{
    if (logger->addToRecentLogsQueueFunc)
    {
        logger->addToRecentLogsQueueFunc(logger->recentLogsQueueVoidPtr,
                                         static_cast<uint8_t>(logLevel),
                                         createFormatterFunc,
                                         argsAsTupleVoidPtr);
    }
}
void setLoggerRecentLogsQueue(LoggerSPtr const& logger, AddToRecentLogsQueueFunc * addToRecentLogsQueueFunc, void * recentLogsQueueVoidPtr)
{
    logger->addToRecentLogsQueueFunc = addToRecentLogsQueueFunc;
    logger->recentLogsQueueVoidPtr   = recentLogsQueueVoidPtr;
}
}}}  // namespace hl_logger::internal