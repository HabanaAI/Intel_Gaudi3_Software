/***************************************************************************
 * Copyright (C) 2017 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ****************************************************************************
 */

#pragma once

#include "types_exception.h"
#include "log_manager.h"
#include "api_calls_counter.hpp"

static synStatus handleException(const char* function)
{
    (void)handleException;  // unused
    try { throw; }
    catch (const SynapseStatusException& e) { LOG_CRITICAL(SYN_API, "exception thrown in function: {}, what: {}", function, e.what()); return e.status(); }
    catch (const std::exception& e)         { LOG_CRITICAL(SYN_API, "exception thrown in function: {}, what: {}", function, e.what()); return synFail; }
    catch (...)                             { LOG_CRITICAL(SYN_API, "exception thrown in function: {}", function); return synFail; }
    return synFail;
}

// Status entry-exit calls are used for regular APIs
#define API_ENTRY_STATUS_TIMED()                                                                                                      \
    static ApiCounter& apiCallCounterInternal{ApiCounterRegistry::getInstance().create(__FUNCTION__)};                                \
    CounterIncrementer apiCallCounterIncrementerInternal(apiCallCounterInternal);                                                     \
    TimeTools::StdTime startTime;                                                                                                     \
    if ( !synSingleton::isSynapseInitialized() || _SYN_SINGLETON_INTERNAL->apiStatsIsEnabled()) startTime = TimeTools::timeNow();     \
    API_ENTRY_STATUS()

#define API_EXIT_STATUS_TIMED(status, stat)                                                                            \
    if (status != synSuccess)                                                                                          \
    {                                                                                                                  \
        LOG_SYN_API("status: {}", status);                                                                             \
    }                                                                                                                  \
    {                                                                                                                  \
        auto      synSingleInternal = _SYN_SINGLETON_INTERNAL;                                                         \
        synStatus postApiCallStatus = synSingleInternal->postApiCallExecution();                                       \
        if (postApiCallStatus != synSuccess)                                                                           \
        {                                                                                                              \
            LOG_DEBUG(SYN_API, "postApiCall failed in function: {}", HLLOG_FUNC);                                      \
            LOG_SYN_API("postApiCall failed. status: {}", postApiCallStatus);                                          \
            return postApiCallStatus;                                                                                  \
        }                                                                                                              \
        uint64_t diff = TimeTools::timeFromNs(startTime);                                                              \
        synSingleInternal->collectStat(StatApiPoints::stat, diff);                                                     \
    }                                                                                                                  \
    return status;                                                                                                     \
    }                                                                                                                  \
    catch (...) { return handleException(__FUNCTION__); }

#define API_ENTRY_STATUS()                                                                                             \
    try                                                                                                                \
    {                                                                                                                  \
        synStatus status = _SYN_SINGLETON_INTERNAL->preApiCallExecution(__FUNCTION__);                                 \
        if (status != synSuccess)                                                                                      \
        {                                                                                                              \
            LOG_DEBUG(SYN_API, "preApiCall failed in function: {}", HLLOG_FUNC);                                       \
            return status;                                                                                             \
        }

#define API_EXIT_STATUS(status)                                                                                        \
    synStatus postApiCallStatus = _SYN_SINGLETON_INTERNAL->postApiCallExecution();                                     \
    if (postApiCallStatus != synSuccess)                                                                               \
    {                                                                                                                  \
        LOG_DEBUG(SYN_API, "postApiCall failed in function: {}", HLLOG_FUNC);                                          \
        return postApiCallStatus;                                                                                      \
    }                                                                                                                  \
    return status;                                                                                                     \
    }                                                                                                                  \
    catch (...) { return handleException(__FUNCTION__); }

// Status specific entry-exit calls are used for APIs which requires a speicific return value upon verification failure
#define API_ENTRY_SPECIFIC(failureStatus)                                                                              \
    try                                                                                                                \
    {                                                                                                                  \
        synStatus status = _SYN_SINGLETON_INTERNAL->preApiCallExecution(__FUNCTION__);                                 \
        if (status != synSuccess)                                                                                      \
        {                                                                                                              \
            LOG_DEBUG(SYN_API, "preApiCall failed in function: {}", HLLOG_FUNC);                                       \
            return failureStatus;                                                                                      \
        }

#define API_EXIT_SPECIFIC(status, failReturn)                                                                          \
    synStatus postApiCallStatus = _SYN_SINGLETON_INTERNAL->postApiCallExecution();                                     \
    if (postApiCallStatus != synSuccess)                                                                               \
    {                                                                                                                  \
        LOG_DEBUG(SYN_API, "postApiCall failed in function: {}", HLLOG_FUNC);                                          \
        return failReturn;                                                                                             \
    }                                                                                                                  \
    return status;                                                                                                     \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        handleException(__FUNCTION__);                                                                                 \
        return failReturn;                                                                                             \
    }
