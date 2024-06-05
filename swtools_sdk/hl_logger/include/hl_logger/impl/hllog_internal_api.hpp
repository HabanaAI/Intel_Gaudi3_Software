#pragma once
#include "../hllog_core.hpp"
#include <optional>

// internal api
namespace hl_logger::internal{
inline namespace v1_0{

extern thread_local uint64_t s_threadId;

using TimePoint = std::chrono::system_clock::time_point;

HLLOG_API TimePoint tscToRt(uint64_t tsc);

struct FormattedLazyLogItem
{
    TimePoint   timePoint;
    int         logLevel = HLLOG_LEVEL_OFF;
    uint64_t    tid;
    std::string msg;
};
using FormattedLazyLogItemOptional = std::optional<FormattedLazyLogItem>;
struct LazyLogInfo
{
    std::string loggerName;
    std::function<FormattedLazyLogItemOptional ()> getNextLogItemFunc;
};
using LazyLogInfoVector = std::vector<LazyLogInfo>;
using LazyLogsHandler   = std::function<LazyLogInfoVector ()>;

HLLOG_API ResourceGuard registerLazyLogsHandler(LazyLogsHandler lazyLogsHandler);

struct IFormatter
{
    virtual ~IFormatter() = default;
    virtual std::string format() const = 0;
};

template <class ... TArgs>
struct Formatter : IFormatter
{
    using ArgsAsTuple = std::tuple<TArgs...>;

    template<class ... Args>
    Formatter(Args && ... args)
    : _argsAsTuple(std::forward<Args>(args)...)
    {

    }
    Formatter(void * argsAsTuplePtr)
    : _argsAsTuple(std::move(*reinterpret_cast<ArgsAsTuple*>(argsAsTuplePtr)))
    {

    }
    std::string format() const
    {
        return std::apply([](auto const & ...args){ return fmt::format(args...); }, _argsAsTuple);
    }
private:
    ArgsAsTuple  _argsAsTuple;
};

using CreateFormatterFunc = IFormatter * (void * buffer, unsigned bufferSize, void * argsAsTupleVoidPtr);

HLLOG_API void logLazy(LoggerSPtr const& logger, int logLevel, CreateFormatterFunc * createFormatterFunc, void * argsAsTupleVoidPtr);

using AddToRecentLogsQueueFunc = void (void* recentLogsQueueVoidPtr,
                                       uint8_t logLevel,
                                       CreateFormatterFunc * createFormatterFunc,
                                       void * argsAsTupleVoidPtr);
HLLOG_API void setLoggerRecentLogsQueue(LoggerSPtr const& logger, AddToRecentLogsQueueFunc * addToRecentLogsQueueFunc, void * recentLogsQueueVoidPtr);
} // vXX
}// hl_logger::internal namespace
