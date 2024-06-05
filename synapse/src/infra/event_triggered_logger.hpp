#pragma once


#include <functional>
#include "syn_logging.h"
// REMARK: The module will be enabled by default
#define ENABLE_EVENT_TRIGGER_LOGGER
#ifdef ENABLE_EVENT_TRIGGER_LOGGER
#define ENABLE_PRE_OPER_PRINTOUTS
#endif

#define MAX_VARIABLE_PARAMS (8)

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// define unlikely (taken from defs.h)
#ifdef unlikely
#undef unlikely
#endif
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace etl
{
template<class T>
struct ToStr
{
    using type = T;
};

// Required for temporal char-array (string.c_str()) for storing content into a string,
// instead of holding it as a pointer
// This is required as the string content might change during run, and effect the stored info
template<>
struct ToStr<const char*>
{
    using type = std::string;
};

// Lambda-Function-Wrapper struct
class LambdaWrapper
{
public:
    LambdaWrapper() {};

    template<class T>
    LambdaWrapper(T lambdaFunction)
    {
        // Instead of being strict, we can update code m_lambdaBuffer* size, only in case required
        static_assert(sizeof(T) <= sizeof(m_lambdaBuffer));

        // Constructing (but not calling heap allocation) the lambdaFunction, passing it to m_lambdaBuffer
        new (m_lambdaBuffer) T(std::move(lambdaFunction));

        /* Expecting to be called with the lambda-function [ptr  := m_lambdaBuffer] */
        // Casting input ptr to T* and calling its DTR
        m_deleter = [](void* ptr) { ((T*)ptr)->~T(); };
        // Casting input ptr to T*, De-referencing it for calling its lambda-function
        m_executor = [](const void* ptr) { return (*((T*)ptr))(); };
        // Perform std:move of the lambda-function (m_lambdaBuffer) from src to dst, and construct it (T) for dst
        // Ensures that arguments (like string) will copy the data,
        // and not just the pointer [The string is a class with a pointer to a buffer]
        m_moveCtor = [](void* dst, void* src) { new (dst) T(std::move((T &&)(*((T*)src)))); };

        m_isPreOperLog = false;
    }

    LambdaWrapper(LambdaWrapper const&) = delete;

    // Move-CTR
    LambdaWrapper(LambdaWrapper&& other)
    : m_deleter(other.m_deleter), m_executor(other.m_executor), m_moveCtor(other.m_moveCtor)
    {
        m_moveCtor(m_lambdaBuffer, other.m_lambdaBuffer);
    }

    // DTR - Deletes the lambda-function, that had been constructed and moved to m_lambdaBuffer, by the CTR
    ~LambdaWrapper()
    {
        if (m_deleter != nullptr)
        {
            m_deleter(m_lambdaBuffer);
        }
    }

    // Executing Lambda functionality
    uint64_t operator()() const { return m_executor(m_lambdaBuffer); }

    explicit operator bool() const { return (m_deleter != nullptr); }

    template<class T>
    void setLambdaWrapper(bool isPreOperLog, T lambdaFunction)
    {
        // Instead of being strict, we can update code m_lambdaBuffer* size, only in case required
        static_assert(sizeof(T) <= sizeof(m_lambdaBuffer));

        if (m_deleter != nullptr)
        {
            m_deleter(m_lambdaBuffer);
        }

        // Constructing (but not calling heap allocation) the lambdaFunction, passing it to m_lambdaBuffer
        new (m_lambdaBuffer) T(std::move(lambdaFunction));

        /* Expecting to be called with the lambda-function [ptr  := m_lambdaBuffer] */
        // Casting input ptr to T* and calling its DTR
        m_deleter = [](void* ptr) { ((T*)ptr)->~T(); };
        // Casting input ptr to T*, De-referencing it for calling its lambda-function
        m_executor = [](const void* ptr) { return (*((T*)ptr))(); };
        // Perform std:move of the lambda-function (m_lambdaBuffer) from src to dst, and construct it (T) for dst
        // Ensures that arguments (like string) will copy the data,
        // and not just the pointer [The string is a class with a pointer to a buffer]
        m_moveCtor = [](void* dst, void* src) { new (dst) T(std::move((T &&)(*((T*)src)))); };

        m_isPreOperLog = isPreOperLog;
    }

    bool isPreOper() const { return m_isPreOperLog; };

private:
    /*
        Due to the below inputs, no need to support variadic lambda size

        1) Basic log-line (w/o user's message)      - 24 Bytes
        2) Adding a message                         - Additional 8 Bytes
        3) Message size does not affect lambda size
        4) Static char array parameter              - Additional 8 Bytes
        5) Temporal string parameter                - Additional 32 Bytes
        6) Integer parameter                        - Additional 8 Bytes per a couple (2)
        7) Float parameter                          - Additional 8 Bytes
    */
    static const uint32_t LAMBDA_BUF_SIZE = 128;

    char m_lambdaBuffer[LAMBDA_BUF_SIZE];  // For saving from allocation calls
    bool m_isPreOperLog = false;

    // Define methods for deleting the Lambda-Function, Executing it and Copy its content
    // The last is required when creating a vector of LambdaWrapper-s, for example
    void (*m_deleter)(void*)            = nullptr;
    uint64_t (*m_executor)(const void*) = nullptr;
    void (*m_moveCtor)(void*, void*)    = nullptr;
};
}  // namespace etl

// Event-Trigger-Logger
//  Current Design:
//      Log-Line - Represent a ... log-line
//      Logger   - Stores Log-Line elements
//                 The Logger can be triggered by a single trigger
//      Manager  - A singleton that is in charge of Logger elements
//      * In case a trigger is common between two Loggers, no order will be kept between the two
//
//  Future design modification (if needed):
//      Trigger-Logger - Defines order between Logger items, upon a given trigger
//      Manager        - A singleton that is in charge of Trigger-Logger elements and Logger elements
//      * Requires, on top of notifying the specific Logger, notifying ALL TLs upon a given logging,
//        In addition, printing the loggers will be done "sequentially", upon order kept by the Trigger-Logger
//

enum eEventLoggerLogType
{
    EVENT_LOGGER_LOG_TYPE_FOR_TESTING,
    EVENT_LOGGER_LOG_TYPE_CS_ORDER,
    EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
    EVENT_LOGGER_LOG_TYPE_MAX  // last
};

enum eEventLoggerTriggerType
{
    EVENT_LOGGER_TRIGGER_TYPE_FOR_TESTING,
    EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER,
    EVENT_LOGGER_TRIGGER_TYPE_CHECK_OPCODES,
    EVENT_LOGGER_TRIGGER_TYPE_MAX  // last
};


using EventTriggeredLogLine = etl::LambdaWrapper;


class VectorWithFormatter
{
public:
    using ctx            = fmt::format_context;
    using basicFormatCtx = fmt::basic_format_args<ctx>;

    VectorWithFormatter(const uint32_t* pValues, uint32_t bufferSize, uint8_t valuesPerLine);

    // TBD = add support libfmt support
    std::string getFormatedArgs() const;

private:
    std::vector<uint32_t> m_values;
    uint8_t               m_valuesPerLine;
    uint32_t              m_bufferSize;
};

class EventTriggeredLogger
{
    class LoggerBuffer
    {
        public:
            void init(uint32_t eventLogsAmount)
            {
                m_eventLogLines.resize(eventLogsAmount);
                m_eventLogsAmount = eventLogsAmount;
            };

            void reset()
            {
                m_index            = 0;
                m_isFull           = false;
                m_isLoggingBlocked = false;
            };

            bool acquire()
            {
                if(unlikely(m_isLoggingBlocked))
                {
                    return false;
                }

                m_loggingRefCount++;
                if(m_isLoggingBlocked)
                {
                    m_loggingRefCount--;
                    return false;
                }

                return true;
            };

            void release()
            {
                m_loggingRefCount--;
            };

            bool isEmpty()
            {
                return ((m_index == 0) && (!m_isFull));
            };

            bool isFull()
            {
                return m_isFull;
            };

            void lockForPrint()
            {
                m_isLoggingBlocked = true;

                while (m_loggingRefCount != 0)
                {
                    std::this_thread::yield();
                }
            };

            EventTriggeredLogLine& getLogLine();

            bool print();

        private:
            enum class eLogIdState : bool
            {
                PRE_OPER,
                LOGGED
            };
            using LogIdToLogStateDb = std::unordered_map<uint64_t, eLogIdState>;

            bool _executeAndValidate(LogIdToLogStateDb&     logIdToStateDb,
                                     uint32_t&              numOfPartialLogs,
                                     EventTriggeredLogLine& eventLogLine);

            std::vector<EventTriggeredLogLine>      m_eventLogLines;
            std::atomic<uint32_t>                   m_index = {0};
            // Postpone trigger, during logging
            std::atomic<uint32_t>                   m_loggingRefCount = {0};
            // Logging is blocked while printing
            // This blocking will not be atomic, as for debug we don't expect to have a reason for that,
            // while wanting to have the verification "as light as possible"
            bool                                    m_isLoggingBlocked = false;
            bool                                    m_isFull = false;
            uint32_t                                m_eventLogsAmount = 0;

            std::mutex                              m_mutex;
    };
    static const unsigned NUM_OF_LOGGER_BUFFERS = 2;

    public:
        EventTriggeredLogger(eEventLoggerLogType        logType,
                             eEventLoggerTriggerType    logTriggerType,
                             uint32_t                   cyclicLoggerSize);

        ~EventTriggeredLogger();

        // Wrapping an operation with pre-logging
        uint64_t preOperation(const char* functionName)
        {
            uint64_t logId = m_nextLogId++;
        #if !defined(ENABLE_PRE_OPER_PRINTOUTS)
            return logId;
        #endif

            LoggerBuffer& localLoggerBuffer = _logggingAcquire();

            uint64_t threadId = (uint64_t) pthread_self();

            EventTriggeredLogLine& eventTriggerLogLine = localLoggerBuffer.getLogLine();
            eventTriggerLogLine.setLambdaWrapper(true, [this, threadId, logId, functionName]()
            {
                std::string log = fmt::format("Start logging of {}: Thread 0x{:x} {}",
                                              logId, threadId,
                                              (functionName != nullptr) ? functionName : "");
                _logCall(log);
                return logId;
            });

            _loggingRelease(localLoggerBuffer);

            return logId;
        }

        template<typename ... Args>
        void addLoggingV(uint64_t           logId,
                         const char*        formattedLog,
                         Args&&...          vArgs)
        {
            LoggerBuffer& localLoggerBuffer = _logggingAcquire();

            uint64_t               threadId            = (uint64_t) pthread_self();

            EventTriggeredLogLine& eventTriggerLogLine = localLoggerBuffer.getLogLine();
            eventTriggerLogLine.setLambdaWrapper(false, [this, threadId, logId, formattedLog, vArgs...]()
            {
                std::string log = fmt::format("End logging of {}: Thread 0x{:x} ", logId, threadId);
                log += fmt::format(formattedLog, vArgs...);

                _logCall(log);
                return logId;
            });

            _loggingRelease(localLoggerBuffer);
        }

        void addLoggingBuffer(uint64_t        logId,
                              const uint32_t* pCommandsBuffer,
                              uint64_t        bufferSize,
                              uint8_t         numOfWordsInLine);

        bool trigger(eEventLoggerTriggerType logTriggerType);

        // Returns true in case same trigger was used
        bool validateTrigger(eEventLoggerTriggerType logTriggerType);

    private:
        bool _print(LoggerBuffer& loggerBuffer);

        void _printHeader();

        static std::string _getLogName(eEventLoggerLogType   logType);

        void _logCall(const std::string& log) const;

        LoggerBuffer& _logggingAcquire()
        {
            unsigned localIndex = m_loggingBufferIndex;

            do
            {
                if (m_loggerBuffers[localIndex].acquire())
                {
                    break;
                }
                toggle(localIndex);
                std::this_thread::yield();
            } while(1);

            return m_loggerBuffers[localIndex];
        };

        void _loggingRelease(LoggerBuffer& loggerBuffer)
        {
            loggerBuffer.release();
        };

        bool _triggerAcquire(LoggerBuffer*& pLocalLoggerBuffer)
        {
            pLocalLoggerBuffer = &m_loggerBuffers[m_triggerBufferIndex];

            if (pLocalLoggerBuffer->isEmpty())
            {  // Nothing to print
                return false;
            }

            pLocalLoggerBuffer->lockForPrint();

            return true;
        };

        void _triggerRelease(LoggerBuffer& localLoggerBuffer)
        {
            localLoggerBuffer.reset();
            toggle(m_triggerBufferIndex);
        };

        void toggle(unsigned& bufferIndex)
        {
            bufferIndex = (1 - bufferIndex);
        };


        eEventLoggerLogType                     m_logType;
        eEventLoggerTriggerType                 m_logTriggerType;

        std::atomic<uint64_t>                   m_nextLogId;
        uint32_t                                m_cyclicLoggerSize;

        LoggerBuffer                            m_loggerBuffers[NUM_OF_LOGGER_BUFFERS];
        unsigned                                m_loggingBufferIndex = 0;
        unsigned                                m_triggerBufferIndex = 0;

        // Single printout at a time
        std::mutex                              m_mutex;

        static const uint64_t UNDEFINED_LOG_ID = std::numeric_limits<uint64_t>::max();
};

class EventTriggeredExecutor
{
    public:
        EventTriggeredExecutor()
        {};

        virtual ~EventTriggeredExecutor()
        {};

        virtual void triggerEventExecution(eEventLoggerTriggerType    logTriggerType) = 0;

        virtual std::string getName() = 0;
};

class EventTriggeredLoggerManager;
typedef std::shared_ptr<EventTriggeredLoggerManager>   spEventTriggeredLoggerManager;

class EventTriggeredLoggerManager
{
    public:
        static const uint32_t MAX_LOGGER_BUFFER_SIZE = 0xFFFFFFFF;

        static spEventTriggeredLoggerManager& getInstance();

        static void releaseInstance();

        EventTriggeredLoggerManager(EventTriggeredLoggerManager const&) = delete;
        void operator=(EventTriggeredLoggerManager const&)  = delete;

        ~EventTriggeredLoggerManager();

        bool createLogger(eEventLoggerLogType        logType,
                          eEventLoggerTriggerType    logTriggerType,
                          uint32_t                   cyclicLoggerSize);

        bool releaseLogger(eEventLoggerLogType      logType);

        void ignoreLogger(eEventLoggerLogType       loggerType);

        bool addExecutor(EventTriggeredExecutor*    pExecutor);

        bool removeExecutor(EventTriggeredExecutor*    pExecutor);

        // For testing - clear all loggers and executors
        void clear();


        uint64_t preOperation(eEventLoggerLogType logType, const char* functionName);

        template<typename ... Args>
        void addLoggingV(eEventLoggerLogType    logType,
                         uint64_t               logId,
                         const char*            formattedLog,
                         Args&&...              vArgs)
        {
            if (m_isEtlDisabled)
            {
                return;
            }

            if (!_isValidLogger(logType))
            {
                return;
            }

            m_eventTriggerLoggers[(uint32_t)logType]->addLoggingV<typename etl::ToStr<Args>::type...>(logId, formattedLog, std::forward<Args>(vArgs)...);
        };

        void addLoggingBuffer(eEventLoggerLogType     logType,
                              uint64_t                logId,
                              const uint32_t*         params,
                              uint64_t                numOfParams,
                              uint8_t                 numOfWordsInLine);

        bool trigger(eEventLoggerTriggerType logTriggerType);

        EventTriggeredLoggerManager();


    private:
        typedef std::shared_ptr<EventTriggeredLogger>                                EventTriggeredLoggerPtr;
        typedef std::array<EventTriggeredLoggerPtr,
                           (uint32_t)eEventLoggerLogType::EVENT_LOGGER_LOG_TYPE_MAX> EventTriggeredLoggerDb;

        typedef std::array<uint8_t,
                           (uint32_t)eEventLoggerLogType::EVENT_LOGGER_LOG_TYPE_MAX> DisabledTriggeredLoggerDb;

        typedef std::deque<EventTriggeredExecutor*>                                  EventTriggeredExecutorDb;

        void _printHeader(eEventLoggerTriggerType   logTriggerType);

        bool _isValidLogger(eEventLoggerLogType    logType);

        static std::string _getLogTriggerName(eEventLoggerTriggerType   logTriggerType);

        EventTriggeredLoggerDb      m_eventTriggerLoggers;
        EventTriggeredExecutorDb    m_eventTriggerExecutors;
        DisabledTriggeredLoggerDb   m_disabledTriggerLogger;

        bool                        m_isEtlDisabled;

        std::mutex                  m_mutex;

        static std::mutex                    m_createMutex;
        static spEventTriggeredLoggerManager m_instance;
};

// TBD - may add file & line
#define SPDL_INFORMATIVE_MSG(msg) "{}: " msg, HLLOG_FUNC
#define ETL_INFORMATIVE_MSG(msg)  "{}: " msg, HLLOG_FUNC

// Functions and Macro definitions for ETL usage
#ifdef ENABLE_EVENT_TRIGGER_LOGGER
inline bool etlTrigger(const eEventLoggerTriggerType trigger)
{
    return EventTriggeredLoggerManager::getInstance()->trigger(trigger);
}

inline uint64_t etlPreOperation(eEventLoggerLogType logType, const char* functionName)
{
    return EventTriggeredLoggerManager::getInstance()->preOperation(logType, functionName);
}

#define ETL_TRIGGER(trigger)                                                                                           \
    etlTrigger(trigger);                                                                                               \

#define ETL_PRE_OPERATION_SET_ID(etlLogId, etlLogType)                                                                 \
    etlLogId = etlPreOperation(etlLogType, __FUNCTION__);

#define ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType)                                                                 \
    uint64_t etlLogId = 0;                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        etlLogId = etlPreOperation(etlLogType, __FUNCTION__);                                                          \
    } while (0)

#define ETL_ADD_LOG(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        EventTriggeredLoggerManager::getInstance()->addLoggingV(etlLogType, etlLogId,                                  \
                                                                ETL_INFORMATIVE_MSG(msg),                              \
                                                                ##__VA_ARGS__);                                        \
        LOG_##spdLogLevel(spdLogId, SPDL_INFORMATIVE_MSG(msg), ##__VA_ARGS__);                                         \
    } while (0)

#define ETL_ADD_LOG_T(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        EventTriggeredLoggerManager::getInstance()->addLoggingV(etlLogType, etlLogId,                                  \
                                                                ETL_INFORMATIVE_MSG(msg),                              \
                                                                ##__VA_ARGS__);                                        \
        LOG_##spdLogLevel##_T(spdLogId, SPDL_INFORMATIVE_MSG(msg), ##__VA_ARGS__);                                     \
    } while (0)

#define ETL_ADD_ETL_ONLY_LOG(etlLogType, etlLogId, msg, ...)                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        EventTriggeredLoggerManager::getInstance()->addLoggingV(etlLogType, etlLogId,                                  \
                                                                ETL_INFORMATIVE_MSG(msg),                              \
                                                                ##__VA_ARGS__);                                        \
    } while (0)

#define ETL_PRINT_BUFFER(etlLogType, etlLogId, buffer, bufferSize, numOfWordsInLine)                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        EventTriggeredLoggerManager::getInstance()->addLoggingBuffer(etlLogType, etlLogId, buffer,                     \
                                                                     bufferSize, numOfWordsInLine);                    \
    } while (0)

#else  // #ifdef ENABLE_EVENT_TRIGGER_LOGGER
#define ETL_TRIGGER(trigger)

#define ETL_PRE_OPERATION_SET_ID(etlLogId, etlLogType)

// For compilations reasons only, we will define and "unuse" the logId parameter
#define ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType)                                                                 \
    uint64_t etlLogId = 0;                                                                                             \
    ((void)etlLogId);

#define ETL_ADD_LOG(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_##spdLogLevel(spdLogId, SPDL_INFORMATIVE_MSG(msg), ##__VA_ARGS__);                                         \
    } while (0)

#define ETL_ADD_LOG_T(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_##spdLogLevel##_T(spdLogId, SPDL_INFORMATIVE_MSG(msg), ##__VA_ARGS__);                                     \
    } while (0)

#define ETL_ADD_ETL_ONLY_LOG(etlLogType, etlLogId, msg, ...)

#define ETL_PRINT_BUFFER(etlLogType, etlLogId, buffer, bufferSize, numOfWordsInLine)

#endif

// One-liners to perform both pre and post ETL-Logging operations
// Start
//      Set ID
#define ETL_ADD_LOG_SET_ID(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                      \
    ETL_PRE_OPERATION_SET_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_LOG(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ##__VA_ARGS__);

#define ETL_ADD_LOG_T_SET_ID(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                    \
    ETL_PRE_OPERATION_SET_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_LOG_T(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ##__VA_ARGS__);

#define ETL_ADD_ETL_ONLY_LOG_SET_ID(etlLogType, etlLogId, msg, ...)                                                    \
    ETL_PRE_OPERATION_SET_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_ETL_ONLY_LOG(etlLogType, etlLogId, msg, ##__VA_ARGS__);

//      Basic - ETL log ID is not required
#define ETL_ADD_LOG_BASIC(etlLogType, spdLogLevel, spdLogId, msg, ...)                                                 \
    ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_LOG(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ##__VA_ARGS__);

#define ETL_ADD_LOG_T_BASIC(etlLogType, spdLogLevel, spdLogId, msg, ...)                                               \
    ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_LOG_T(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ##__VA_ARGS__);

#define ETL_ADD_ETL_ONLY_LOG_BASIC(etlLogType, msg, ...)                                                               \
    ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_ETL_ONLY_LOG(etlLogType, etlLogId, msg, ##__VA_ARGS__);

//      New ID
#define ETL_ADD_LOG_NEW_ID(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                      \
    ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_LOG(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ##__VA_ARGS__);

#define ETL_ADD_LOG_T_NEW_ID(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ...)                                    \
    ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_LOG_T(etlLogType, etlLogId, spdLogLevel, spdLogId, msg, ##__VA_ARGS__);

#define ETL_ADD_ETL_ONLY_LOG_NEW_ID(etlLogType, etlLogId, msg, ...)                                                    \
    ETL_PRE_OPERATION_NEW_ID(etlLogId, etlLogType);                                                                    \
    ETL_ADD_ETL_ONLY_LOG(etlLogType, etlLogId, msg, ##__VA_ARGS__);
// End

// Log-Levels support
//      ETL_ADD_LOG
#define ETL_ADD_LOG_TRACE(etlLogType, etlLogId, spdLogId, msg, ...)                                                    \
    ETL_ADD_LOG(etlLogType, etlLogId, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_DEBUG(etlLogType, etlLogId, spdLogId, msg, ...)                                                    \
    ETL_ADD_LOG(etlLogType, etlLogId, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_INFO(etlLogType, etlLogId, spdLogId, msg, ...)                                                     \
    ETL_ADD_LOG(etlLogType, etlLogId, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_WARN(etlLogType, etlLogId, spdLogId, msg, ...)                                                     \
    ETL_ADD_LOG(etlLogType, etlLogId, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_ERR(etlLogType, etlLogId, spdLogId, msg, ...)                                                      \
    ETL_ADD_LOG(etlLogType, etlLogId, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_CRITICAL(etlLogType, etlLogId, spdLogId, msg, ...)                                                 \
    ETL_ADD_LOG(etlLogType, etlLogId, CRITICAL, spdLogId, msg, ##__VA_ARGS__)

#define ETL_ADD_LOG_T_TRACE(etlLogType, etlLogId, spdLogId, msg, ...)                                                  \
    ETL_ADD_LOG_T(etlLogType, etlLogId, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_DEBUG(etlLogType, etlLogId, spdLogId, msg, ...)                                                  \
    ETL_ADD_LOG_T(etlLogType, etlLogId, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_INFO(etlLogType, etlLogId, spdLogId, msg, ...)                                                   \
    ETL_ADD_LOG_T(etlLogType, etlLogId, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_WARN(etlLogType, etlLogId, spdLogId, msg, ...)                                                   \
    ETL_ADD_LOG_T(etlLogType, etlLogId, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_ERR(etlLogType, etlLogId, spdLogId, msg, ...)                                                    \
    ETL_ADD_LOG_T(etlLogType, etlLogId, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_CRITICAL(etlLogType, etlLogId, spdLogId, msg, ...)                                               \
    ETL_ADD_LOG_T(etlLogType, etlLogId, CRITICAL, spdLogId, msg, ##__VA_ARGS__)

#define ETL_ADD_LOG_SET_ID_TRACE(etlLogType, etlLogId, spdLogId, msg, ...)                                             \
    ETL_ADD_LOG_SET_ID(etlLogType, etlLogId, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_SET_ID_DEBUG(etlLogType, etlLogId, spdLogId, msg, ...)                                             \
    ETL_ADD_LOG_SET_ID(etlLogType, etlLogId, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_SET_ID_INFO(etlLogType, etlLogId, spdLogId, msg, ...)                                              \
    ETL_ADD_LOG_SET_ID(etlLogType, etlLogId, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_SET_ID_WARN(etlLogType, etlLogId, spdLogId, msg, ...)                                              \
    ETL_ADD_LOG_SET_ID(etlLogType, etlLogId, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_SET_ID_ERR(etlLogType, etlLogId, spdLogId, msg, ...)                                               \
    ETL_ADD_LOG_SET_ID(etlLogType, etlLogId, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_SET_ID_CRITICAL(etlLogType, etlLogId, spdLogId, msg, ...)                                          \
    ETL_ADD_LOG_SET_ID(etlLogType, etlLogId, CRITICAL, spdLogId, msg, ##__VA_ARGS__)

#define ETL_ADD_LOG_T_SET_ID_TRACE(etlLogType, etlLogId, spdLogId, msg, ...)                                           \
    ETL_ADD_LOG_T_SET_ID(etlLogType, etlLogId, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_SET_ID_DEBUG(etlLogType, etlLogId, spdLogId, msg, ...)                                           \
    ETL_ADD_LOG_T_SET_ID(etlLogType, etlLogId, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_SET_ID_INFO(etlLogType, etlLogId, spdLogId, msg, ...)                                            \
    ETL_ADD_LOG_T_SET_ID(etlLogType, etlLogId, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_SET_ID_WARN(etlLogType, etlLogId, spdLogId, msg, ...)                                            \
    ETL_ADD_LOG_T_SET_ID(etlLogType, etlLogId, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_SET_ID_ERR(etlLogType, etlLogId, spdLogId, msg, ...)                                             \
    ETL_ADD_LOG_T_SET_ID(etlLogType, etlLogId, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_SET_ID_CRITICAL(etlLogType, etlLogId, spdLogId, msg, ...)                                        \
    ETL_ADD_LOG_T_SET_ID(etlLogType, etlLogId, CRITICAL, spdLogId, msg, ##__VA_ARGS__)

#define ETL_ADD_LOG_NEW_ID_TRACE(etlLogType, etlLogId, spdLogId, msg, ...)                                             \
    ETL_ADD_LOG_NEW_ID(etlLogType, etlLogId, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_NEW_ID_DEBUG(etlLogType, etlLogId, spdLogId, msg, ...)                                             \
    ETL_ADD_LOG_NEW_ID(etlLogType, etlLogId, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_NEW_ID_INFO(etlLogType, etlLogId, spdLogId, msg, ...)                                              \
    ETL_ADD_LOG_NEW_ID(etlLogType, etlLogId, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_NEW_ID_WARN(etlLogType, etlLogId, spdLogId, msg, ...)                                              \
    ETL_ADD_LOG_NEW_ID(etlLogType, etlLogId, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_NEW_ID_ERR(etlLogType, etlLogId, spdLogId, msg, ...)                                               \
    ETL_ADD_LOG_NEW_ID(etlLogType, etlLogId, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_NEW_ID_CRITICAL(etlLogType, etlLogId, spdLogId, msg, ...)                                          \
    ETL_ADD_LOG_NEW_ID(etlLogType, etlLogId, CRITICAL, spdLogId, msg, ##__VA_ARGS__)

#define ETL_ADD_LOG_T_NEW_ID_TRACE(etlLogType, etlLogId, spdLogId, msg, ...)                                           \
    ETL_ADD_LOG_T_NEW_ID(etlLogType, etlLogId, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_NEW_ID_DEBUG(etlLogType, etlLogId, spdLogId, msg, ...)                                           \
    ETL_ADD_LOG_T_NEW_ID(etlLogType, etlLogId, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_NEW_ID_INFO(etlLogType, etlLogId, spdLogId, msg, ...)                                            \
    ETL_ADD_LOG_T_NEW_ID(etlLogType, etlLogId, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_NEW_ID_WARN(etlLogType, etlLogId, spdLogId, msg, ...)                                            \
    ETL_ADD_LOG_T_NEW_ID(etlLogType, etlLogId, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_NEW_ID_ERR(etlLogType, etlLogId, spdLogId, msg, ...)                                             \
    ETL_ADD_LOG_T_NEW_ID(etlLogType, etlLogId, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ADD_LOG_T_NEW_ID_CRITICAL(etlLogType, etlLogId, spdLogId, msg, ...)                                        \
    ETL_ADD_LOG_T_NEW_ID(etlLogType, etlLogId, CRITICAL, spdLogId, msg, ##__VA_ARGS__)

#define ETL_TRACE(etlLogType, spdLogId, msg, ...)                                                                      \
    ETL_ADD_LOG_BASIC(etlLogType, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_DEBUG(etlLogType, spdLogId, msg, ...)                                                                      \
    ETL_ADD_LOG_BASIC(etlLogType, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_INFO(etlLogType, spdLogId, msg, ...)                                                                       \
    ETL_ADD_LOG_BASIC(etlLogType, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_WARN(etlLogType, spdLogId, msg, ...)                                                                       \
    ETL_ADD_LOG_BASIC(etlLogType, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_ERR(etlLogType, spdLogId, msg, ...)                                                                        \
    ETL_ADD_LOG_BASIC(etlLogType, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_CRITICAL(etlLogType, spdLogId, msg, ...)                                                                   \
    ETL_ADD_LOG_BASIC(etlLogType, CRITICAL, spdLogId, msg, ##__VA_ARGS__)

#define ETL_T_TRACE(etlLogType, spdLogId, msg, ...)                                                                    \
    ETL_ADD_LOG_T_BASIC(etlLogType, TRACE, spdLogId, msg, ##__VA_ARGS__)
#define ETL_T_DEBUG(etlLogType, spdLogId, msg, ...)                                                                    \
    ETL_ADD_LOG_T_BASIC(etlLogType, DEBUG, spdLogId, msg, ##__VA_ARGS__)
#define ETL_T_INFO(etlLogType, spdLogId, msg, ...)                                                                     \
    ETL_ADD_LOG_T_BASIC(etlLogType, INFO, spdLogId, msg, ##__VA_ARGS__)
#define ETL_T_WARN(etlLogType, spdLogId, msg, ...)                                                                     \
    ETL_ADD_LOG_T_BASIC(etlLogType, WARN, spdLogId, msg, ##__VA_ARGS__)
#define ETL_T_ERR(etlLogType, spdLogId, msg, ...)                                                                      \
    ETL_ADD_LOG_T_BASIC(etlLogType, ERR, spdLogId, msg, ##__VA_ARGS__)
#define ETL_T_CRITICAL(etlLogType, spdLogId, msg, ...)                                                                 \
    ETL_ADD_LOG_T_BASIC(etlLogType, CRITICAL, spdLogId, msg, ##__VA_ARGS__)
