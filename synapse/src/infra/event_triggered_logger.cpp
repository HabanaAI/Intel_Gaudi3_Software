#include "event_triggered_logger.hpp"

#include "defs.h"
#include "habana_global_conf.h"
#include "log_manager.h"
#include "types_exception.h"

#include <algorithm>
#include <unistd.h>

#define SPDL_LOG_HDR  LOG_CRITICAL
#define SPDL_LOG_LINE LOG_ERR

std::mutex                    EventTriggeredLoggerManager::m_createMutex;
spEventTriggeredLoggerManager EventTriggeredLoggerManager::m_instance = nullptr;

// The max amount of parameters, when using the API which supports adding variable amount of them

// --- VectorWithFormatter  --- //
VectorWithFormatter ::VectorWithFormatter(const uint32_t* pValues, uint32_t bufferSize, uint8_t valuesPerLine)
: m_values((uint32_t*)pValues, (uint32_t*)pValues + ((bufferSize + 1) / sizeof(uint32_t))),
  m_valuesPerLine(valuesPerLine),
  m_bufferSize(bufferSize)
{
    if ((valuesPerLine == 0) || (valuesPerLine > MAX_VARIABLE_PARAMS) || (bufferSize % sizeof(uint32_t) != 0))
    {
        throw SynapseException("Invalid valuesPerLine");
    }
}

std::string VectorWithFormatter ::getFormatedArgs() const
{
    std::string formattedStr;
    for (auto it = m_values.begin(); it != m_values.end();)
    {
        unsigned lineLen   = std::min(unsigned(m_valuesPerLine), unsigned(m_values.end() - it));
        auto     lineEndIt = std::next(it, lineLen);
        formattedStr += fmt::format("{:>010x}", fmt::join(it, lineEndIt, " "));
        if (lineEndIt != m_values.end())
        {
            formattedStr += "\n";
        }
        it = lineEndIt;
    }

    return formattedStr;
}

// --- LoggerBuffer --- //
EventTriggeredLogLine& EventTriggeredLogger::LoggerBuffer::getLogLine()
{
    uint32_t logIndex = m_index++;

    // In case buffer is very small, we may want to repeat this
    while (unlikely(logIndex >= m_eventLogsAmount))
    {
        m_isFull = true;

        // Only one thread can reset
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_index >= m_eventLogsAmount)
            {
                m_index = 0;
            }
        }

        // Re-take a valid index
        logIndex = m_index++;
    }

    return m_eventLogLines[logIndex];
}

bool EventTriggeredLogger::LoggerBuffer::print()
{
    bool status = true;

    // Log-ID to state map
    LogIdToLogStateDb logIdToStateDb;

    // LOG-IDs that had pre-oper calls but no logging
    uint32_t numOfPartialLogs = 0;

    if (isFull())
    {
        for (uint32_t i = m_index; i < m_eventLogsAmount; i++)
        {
            EventTriggeredLogLine& eventLogLine = m_eventLogLines[i];
            if (!_executeAndValidate(logIdToStateDb, numOfPartialLogs, eventLogLine))
            {
                status = false;
            }
        }
    }

    for (uint32_t i = 0; i < m_index; i++)
    {
        EventTriggeredLogLine& eventLogLine = m_eventLogLines[i];
        if (!_executeAndValidate(logIdToStateDb, numOfPartialLogs, eventLogLine))
        {
            status = false;
        }
    }

    SPDL_LOG_LINE(EVENT_TRIGGER, "numOfPartialLogs {}", numOfPartialLogs);

    return status;
}

bool EventTriggeredLogger::LoggerBuffer::_executeAndValidate(LogIdToLogStateDb&     logIdToStateDb,
                                                             uint32_t&              numOfPartialLogs,
                                                             EventTriggeredLogLine& eventLogLine)
{
    if (eventLogLine)
    {
        uint64_t currentLogId = eventLogLine();
        bool     isPreOper    = eventLogLine.isPreOper();

        auto logIdToStateEndIter = logIdToStateDb.end();
        auto logIdToStateIter    = logIdToStateDb.find(currentLogId);

        if (logIdToStateIter != logIdToStateEndIter)
        {
            if (logIdToStateIter->second == eLogIdState::PRE_OPER)
            {
                if (isPreOper)
                {
                    LOG_CRITICAL(EVENT_TRIGGER, "Log {} has double pre-oper calls", currentLogId);
                    return false;
                }
            }
            else
            {
                if (!isPreOper)
                {
                    LOG_CRITICAL(EVENT_TRIGGER, "Log {} had been already logged", currentLogId);
                    return false;
                }
            }
        }

        if (isPreOper)
        {
            logIdToStateDb[currentLogId] = eLogIdState::PRE_OPER;
            numOfPartialLogs++;
        }
        else
        {
            logIdToStateDb[currentLogId] = eLogIdState::LOGGED;
            numOfPartialLogs--;
        }
    }

    return true;
}

// --- EventTriggeredLogger --- //
EventTriggeredLogger::EventTriggeredLogger(eEventLoggerLogType     logType,
                                           eEventLoggerTriggerType logTriggerType,
                                           uint32_t                cyclicLoggerSize)
: m_logType(logType), m_logTriggerType(logTriggerType), m_nextLogId(0), m_cyclicLoggerSize(cyclicLoggerSize)
{
    m_loggerBuffers[0].init(cyclicLoggerSize);
    m_loggerBuffers[1].init(cyclicLoggerSize);
}

EventTriggeredLogger::~EventTriggeredLogger() {}

void EventTriggeredLogger::addLoggingBuffer(uint64_t        logId,
                                            const uint32_t* pCommandsBuffer,
                                            uint64_t        bufferSize,
                                            uint8_t         numOfWordsInLine)
{
    LoggerBuffer& localLoggerBuffer = _logggingAcquire();

    uint64_t threadId = (uint64_t)pthread_self();

    EventTriggeredLogLine& eventTriggerLogLine = localLoggerBuffer.getLogLine();
    try
    {
        VectorWithFormatter variadicVec(pCommandsBuffer, bufferSize, numOfWordsInLine);

        eventTriggerLogLine.setLambdaWrapper(false, [this, threadId, logId, variadicVec]() {
            std::string log = fmt::format("End logging of {}: Thread 0x{:x}:\n", logId, threadId);
            log += variadicVec.getFormatedArgs();

            _logCall(log);
            return logId;
        });
    }
    catch (const SynapseException&)
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Failed to add log to buffer");
    }

    _loggingRelease(localLoggerBuffer);
}

bool EventTriggeredLogger::trigger(eEventLoggerTriggerType logTriggerType)
{
    if (m_logTriggerType != logTriggerType)
    {
        return true;
    }

    // Ensure that only one trigger is in-progress
    std::unique_lock<std::mutex> lock(m_mutex);
    LoggerBuffer*                pLocalLoggerBuffer = nullptr;

    if (!_triggerAcquire(pLocalLoggerBuffer))
    {
        // Nothing to print
        return true;
    }

    HB_ASSERT_PTR(pLocalLoggerBuffer);
    LoggerBuffer& localLoggerBuffer = *pLocalLoggerBuffer;

    bool status = _print(localLoggerBuffer);

    _triggerRelease(localLoggerBuffer);

    return status;
}

bool EventTriggeredLogger::validateTrigger(eEventLoggerTriggerType logTriggerType)
{
    return (m_logTriggerType == logTriggerType);
}

bool EventTriggeredLogger::_print(LoggerBuffer& loggerBuffer)
{
    bool status = true;

    _printHeader();
    status = loggerBuffer.print();

    return status;
}

void EventTriggeredLogger::_printHeader()
{
    std::string    basicHeader("Event-Triggered logger");
    const uint32_t headerBlockPadding = 10;
    std::string    name               = _getLogName(m_logType);
    uint32_t       headerBlockSize    = std::max(name.size(), basicHeader.size()) + headerBlockPadding;

    std::string header = fmt::format("\n^{0:-^{3}}^\n"
                                     "^{1: ^{3}}^\n"
                                     "^{2: ^{3}}^\n"
                                     "^{0:-^{3}}^\n",
                                     "",
                                     basicHeader,
                                     name,
                                     headerBlockSize);

    SPDL_LOG_HDR(EVENT_TRIGGER, "{}", header);
}

std::string EventTriggeredLogger::_getLogName(eEventLoggerLogType   logType)
{
    switch (logType)
    {
        case EVENT_LOGGER_LOG_TYPE_CS_ORDER:
            return "default";

        case EVENT_LOGGER_LOG_TYPE_FOR_TESTING:
           return "for testing";

        case EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES:
            return "Check OPCODE";

        case EVENT_LOGGER_LOG_TYPE_MAX:
            break;
    }

    return "Invalid";
}

void EventTriggeredLogger::_logCall(const std::string& log) const
{
    SPDL_LOG_LINE(EVENT_TRIGGER, "{}", log);
}

// --- EventTriggeredLoggerManager --- //
spEventTriggeredLoggerManager& EventTriggeredLoggerManager::getInstance()
{
    if (m_instance == nullptr)
    {
        std::unique_lock<std::mutex> lock(m_createMutex);
        if (m_instance == nullptr)
        {
            m_instance = std::make_shared<EventTriggeredLoggerManager>();
        }
    }

    return m_instance;
}

void EventTriggeredLoggerManager::releaseInstance()
{
    m_instance = nullptr;
}

EventTriggeredLoggerManager::EventTriggeredLoggerManager()
{
    // we would like to have the option to see this construction low-level log-line
    LOG_TRACE(EVENT_TRIGGER, "EventTriggeredLoggerManager constructed");

    m_isEtlDisabled = GCFG_ETL_DISABLE.value();

    clear();
}

EventTriggeredLoggerManager::~EventTriggeredLoggerManager()
{
}

bool EventTriggeredLoggerManager::createLogger(eEventLoggerLogType        logType,
                                               eEventLoggerTriggerType    logTriggerType,
                                               uint32_t                   cyclicLoggerSize)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if (unlikely((logType >= EVENT_LOGGER_LOG_TYPE_MAX               ) ||
                 (m_eventTriggerLoggers[(uint32_t)logType] != nullptr)))
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Logger {} had already been defined or is invalid", logType);
        return false;
    }

    if (cyclicLoggerSize >= MAX_LOGGER_BUFFER_SIZE)
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Logger {} got invalid buffer-size (0x{:x})", logType, cyclicLoggerSize);
        return false;
    }

    EventTriggeredLoggerPtr eventTriggeredLogger =
                                std::make_shared<EventTriggeredLogger>(logType, logTriggerType, cyclicLoggerSize);

    m_eventTriggerLoggers[(uint32_t)logType] = eventTriggeredLogger;

    m_disabledTriggerLogger[(uint32_t)logType] = false;

    return true;
}

bool EventTriggeredLoggerManager::releaseLogger(eEventLoggerLogType logType)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if (unlikely((logType >= EVENT_LOGGER_LOG_TYPE_MAX) || (m_eventTriggerLoggers[(uint32_t)logType] == nullptr)))
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Logger {} is not defined", logType);
        return false;
    }

    m_eventTriggerLoggers[(uint32_t)logType] = nullptr;

    m_disabledTriggerLogger[(uint32_t)logType] = false;

    return true;
}

void EventTriggeredLoggerManager::ignoreLogger(eEventLoggerLogType loggerType)
{
    m_disabledTriggerLogger[(uint32_t)loggerType] = true;
}

bool EventTriggeredLoggerManager::addExecutor(EventTriggeredExecutor* pExecutor)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if (std::find(m_eventTriggerExecutors.begin(), m_eventTriggerExecutors.end(), pExecutor) !=
        m_eventTriggerExecutors.end())
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Executor {} was already added to the DB", pExecutor->getName());
        return false;
    }

    m_eventTriggerExecutors.push_back(pExecutor);
    return true;
}

bool EventTriggeredLoggerManager::removeExecutor(EventTriggeredExecutor* pExecutor)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    auto pExecutorItr = std::find(m_eventTriggerExecutors.begin(), m_eventTriggerExecutors.end(), pExecutor);
    if (pExecutorItr == m_eventTriggerExecutors.end())
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Executor {} was not found in DB", pExecutor->getName());
        return false;
    }

    m_eventTriggerExecutors.erase(pExecutorItr);
    return true;
}

void EventTriggeredLoggerManager::clear()
{
    m_eventTriggerExecutors.clear();

    for (unsigned i = 0; i < eEventLoggerLogType::EVENT_LOGGER_LOG_TYPE_MAX; i++)
    {
        m_eventTriggerLoggers[i]   = nullptr;
        m_disabledTriggerLogger[i] = false;
    }
}

uint64_t EventTriggeredLoggerManager::preOperation(eEventLoggerLogType logType, const char* functionName)
{
    if (m_isEtlDisabled)
    {
        return 0;
    }

    if (!_isValidLogger(logType))
    {
        return MAX_LOGGER_BUFFER_SIZE;
    }

    if (unlikely((logType >= EVENT_LOGGER_LOG_TYPE_MAX               ) ||
                 (m_eventTriggerLoggers[(uint32_t)logType] == nullptr)))
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Logger {} has not been defined or is invalid", logType);
        return MAX_LOGGER_BUFFER_SIZE;
    }

    return m_eventTriggerLoggers[(uint32_t)logType]->preOperation(functionName);
}

void EventTriggeredLoggerManager::addLoggingBuffer(eEventLoggerLogType    logType,
                                                   uint64_t               logId,
                                                   const uint32_t*        params,
                                                   uint64_t               bufferSize,
                                                   uint8_t                numOfWordsInLine)
{
    if (m_isEtlDisabled)
    {
        return;
    }

    if (!_isValidLogger(logType))
    {
        return;
    }

    m_eventTriggerLoggers[(uint32_t)logType]->addLoggingBuffer(logId, params, bufferSize, numOfWordsInLine);
}

bool EventTriggeredLoggerManager::trigger(eEventLoggerTriggerType logTriggerType)
{
    _printHeader(logTriggerType);

    bool status = true;

    for (auto eventTriggerLogger : m_eventTriggerLoggers)
    {
        if (eventTriggerLogger != nullptr)
        {
            if (!(eventTriggerLogger->trigger(logTriggerType)))
            {
                status = false;
            }
        }
    }

    for (auto pExecutor : m_eventTriggerExecutors)
    {
        SPDL_LOG_HDR(EVENT_TRIGGER,
                     "Calling Event-Triggered-Executor \"{}\" (0x{:x}):",
                     pExecutor->getName(),
                     (uint64_t)pExecutor);

        pExecutor->triggerEventExecution(logTriggerType);
    }

    return status;
}

void EventTriggeredLoggerManager::_printHeader(eEventLoggerTriggerType   logTriggerType)
{
    std::string    basicHeader("Event-Triggered logger manager");
    const uint32_t headerBlockPadding = 10;
    std::string    description        = _getLogTriggerName(logTriggerType);
    uint32_t       headerBlockSize    = std::max(description.size(), basicHeader.size()) + headerBlockPadding;

    std::string header = fmt::format("\n^{0:-^{3}}^\n"
                                     "^{1: ^{3}}^\n"
                                     "^{2: ^{3}}^\n"
                                     "^{0:-^{3}}^\n",
                                     "",
                                     basicHeader,
                                     description,
                                     headerBlockSize);

    SPDL_LOG_HDR(EVENT_TRIGGER, "{}", header);
}

std::string EventTriggeredLoggerManager::_getLogTriggerName(eEventLoggerTriggerType   logTriggerType)
{
    switch (logTriggerType)
    {
        case EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER:
            return "default trigger";

        case EVENT_LOGGER_TRIGGER_TYPE_FOR_TESTING:
            return "for-testing trigger";

        case EVENT_LOGGER_TRIGGER_TYPE_CHECK_OPCODES:
            return "check-opcode trigger";

        case EVENT_LOGGER_TRIGGER_TYPE_MAX:
            break;
    }

    return "Invalid";
}

// returns returnType in case logType is invalid, not exists or disabled
bool EventTriggeredLoggerManager::_isValidLogger(eEventLoggerLogType logType)
{
    if (unlikely(logType >= EVENT_LOGGER_LOG_TYPE_MAX))
    {
        LOG_CRITICAL(EVENT_TRIGGER, "LoggerType {} is invalid", logType);
        return false;
    }

    if (unlikely(m_eventTriggerLoggers[(uint32_t)logType] == nullptr))
    {
        LOG_CRITICAL(EVENT_TRIGGER, "Logger {} is invalid", logType);
        return false;
    }

    if (unlikely(m_disabledTriggerLogger[(uint32_t)logType]))
    {
        return false;
    }

    return true;
}