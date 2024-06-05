#include "scal_event.hpp"

#include "hlthunk.h"
#include "log_manager.h"
#include "synapse_types.h"
#include "stream_base_scal.hpp"
#include "habana_global_conf_runtime.h"

#include <sys/mman.h>
#include <unistd.h>

#define TIMESTAMP_READ_RETRY_VALUE 1000000
#define TIMESTAMP_READ_SLEEP_VALUE 100000 // 100 msec

ScalEvent::ScalEvent(uint32_t devIdx, const unsigned int flags, TimestampBuff* timestampBuffer)
: timestampBuff(timestampBuffer)
{
    collectTime = (flags & EVENT_COLLECT_TIME) != 0;
    if (collectTime && !timestampBuff)
    {
        LOG_ERR(SYN_STREAM, "ScalEvent of type EVENT_COLLECT_TIME must have TimestampBuff");
        return;
    }
}

ScalEvent::ScalEvent(const ScalEvent& other)
{
    std::lock_guard guard(other);

    longSo          = other.longSo;
    pStreamIfScal   = other.pStreamIfScal;
    m_isOnHclStream = other.m_isOnHclStream;
    hclSyncInfo     = other.hclSyncInfo;
    eventHandle     = other.eventHandle;
    collectTime     = other.collectTime;
    timestampBuff   = other.timestampBuff;
    m_waitMode      = other.m_waitMode;
}


void ScalEvent::clearState()
{
    EventWithMappedTensor::clearState();
    longSo          = {};
    pStreamIfScal   = nullptr;
    hclSyncInfo     = {};
    m_isOnHclStream = false;
    if (timestampBuff)
    {
        *timestampBuff->timestamp = 0;
    }
}

void ScalEvent::setMappedTensor(uint64_t              tensorOffset,
                                uint32_t              tensorId,
                                const char*           tensorName,
                                InternalRecipeHandle* pInternalRecipeHandle)
{
    EventWithMappedTensor::setMappedTensor(tensorOffset, tensorId, tensorName, pInternalRecipeHandle);
    m_isOnHclStream = false;
}

void ScalEvent::lock() const
{
    m_mutex.lock();
}

void ScalEvent::unlock() const
{
    m_mutex.unlock();
}


void ScalEvent::setOnHclStream(bool onHclStream /* = true*/)
{
    m_isOnHclStream = onHclStream;
    if (isOnHclStream())
    {
        clearTensorMapping();
    }
}

synStatus ScalEvent::getTime(uint64_t& nanoseconds, bool start) const
{
    std::string msg = start ? "Start" : "End";
    if (!collectTime)
    {
        LOG_ERR_T(SYN_STREAM, "{}: {} Event {} is not collecting time", HLLOG_FUNC, msg, toString());
        return synInvalidArgument;
    }
    if (!isOnHclStream() & !pStreamIfScal)
    {
        LOG_ERR_T(SYN_STREAM, "{}: {} Event {} was not recorded", HLLOG_FUNC, msg, toString());
        return synObjectNotInitialized;
    }
    if (timestampBuff == nullptr)
    {
        LOG_ERR_T(SYN_STREAM,
                  "{}: {} Event {} is collecting time but doesn't have timestampBuff",
                  HLLOG_FUNC,
                  msg,
                  toString());
        return synInvalidArgument;
    }
    if ((pStreamIfScal && pStreamIfScal->eventQuery(*this) == synSuccess) ||
        (isOnHclStream() &&
         scal_completion_group_wait(hclSyncInfo.cp_handle, hclSyncInfo.targetValue, 0) == SCAL_SUCCESS))
    {
        // Busy wait until timestamp value filled by LKD thread
        // First try to read with a retry counter
        unsigned timestampReadRetryCounter = TIMESTAMP_READ_RETRY_VALUE;
        while (0 < timestampReadRetryCounter)
        {
            if (*timestampBuff->timestamp == 0ULL)
            {
                timestampReadRetryCounter--;
                continue;
            }
            break;
        }
        if (!timestampReadRetryCounter)
        {
            LOG_DEBUG(SYN_STREAM,
                      "{}: {} {} Event {} timestamp is not valid after {} read retries",
                      HLLOG_FUNC,
                      msg,
                      isOnHclStream() ? "HCL" : "Synapse",
                      toString(),
                      TIMESTAMP_READ_RETRY_VALUE);

            // If TIMESTAMP_READ_RETRY_VALUE was not enough and the timestamp is not available,
            // Sleep 1 usec TIMESTAMP_READ_SLEEP_VALUE times until timestamp is available.
            unsigned timestampReadSleepCounter = TIMESTAMP_READ_SLEEP_VALUE;
            while (0 < timestampReadSleepCounter)
            {
                usleep(1);
                if (*timestampBuff->timestamp == 0ULL)
                {
                    timestampReadSleepCounter--;
                    continue;
                }
                break;
            }
            if (!timestampReadSleepCounter)
            {
                LOG_ERR_T(SYN_STREAM,
                          "{}: {} {} Event {} timestamp is not valid after timeout reached",
                          HLLOG_FUNC,
                          msg,
                          isOnHclStream() ? "HCL" : "Synapse",
                          toString());
                return synFail;
            }
        }

        nanoseconds = *timestampBuff->timestamp;
    }
    else
    {
        LOG_DEBUG_T(SYN_STREAM,
                    "{}: {} Event {} did not expire (timestamp {})",
                    HLLOG_FUNC,
                    msg,
                    toString(),
                    *timestampBuff->timestamp);
        return synBusy;
    }
    return synSuccess;
}

std::pair<synEventHandle, SlotMapItemSptr<ScalEvent>> ScalEventsPool::getNewEvent(unsigned int flags)
{
    TimestampBuff* timestampBuff = nullptr;
    if (flags & EVENT_COLLECT_TIME)
    {
        if (acquireNextAvailableTimestamp(timestampBuff) != synSuccess)
        {
            LOG_WARN(SYN_API, "getNewEvent: Failed to create new event - acquireNextAvailableTimestamp failed");
            return {nullptr, nullptr};
        }
    }
    auto result      = m_eventsDB.insert(0, 0, flags, timestampBuff);
    auto resultTyped = std::make_pair((synEventHandle)result.first, std::move(result.second));

    if (resultTyped.second != nullptr)
    {
        resultTyped.second->eventHandle = resultTyped.first;
    }
    else
    {
        LOG_WARN(SYN_API, "getNewEvent: Failed to create new event");
    }
    return resultTyped;
}

synStatus ScalEventsPool::destroyEvent(synEventHandle eventHandle)
{
    auto eventSptr = m_eventsDB[(SMHandle)eventHandle];
    if (eventSptr)
    {
        uint64_t cbOffsetInFd        = 0;
        uint64_t indexInCbOffsetInFd = 0;
        if (eventSptr->timestampBuff)
        {
            cbOffsetInFd        = eventSptr->timestampBuff->cbOffsetInFd;
            indexInCbOffsetInFd = eventSptr->timestampBuff->indexInCbOffsetInFd;
        }
        eventSptr.reset();
        synStatus ret = m_eventsDB.erase((SMHandle)eventHandle) ? synSuccess : synFail;
        if (ret == synSuccess)
        {
            if (cbOffsetInFd)
            {
                // only if deletion succeeded - mark timestamp as free
                std::lock_guard lock(m_mutex);
                m_eventTimeCollectionHandles[cbOffsetInFd].unusedTimestamps[indexInCbOffsetInFd] = 1;
            }
        }
        else
        {
            LOG_ERR(SYN_API,
                    "destroyEvent: failed to destroy event {}, maybe it was already destroyed",
                    (SMHandle)eventHandle);
        }
        return ret;
    }
    LOG_ERR(SYN_API, "Failed to find event {}", (SMHandle)eventHandle);
    return synFail;
}

SlotMapItemSptr<ScalEvent> ScalEventsPool::getEventSptr(synEventHandle eventHandle)
{
    return m_eventsDB[(SMHandle)eventHandle];
}

ScalEventsPool::ScalEventsPool(int fd) : m_fd(fd), m_eventsDB(GCFG_NUM_OF_USER_STREAM_EVENTS.value()) {}

ScalEventsPool::~ScalEventsPool()
{
    // unmap event time collection buffers
    for (auto& eventTimeCollectionHandle : m_eventTimeCollectionHandles)
    {
        unsigned ret = munmap(eventTimeCollectionHandle.second.timestampsMappedMemory,
                              NUMBER_OF_TIME_COLLECTING_EVENTS_PER_HANDLE * sizeof(uint64_t));
        if (ret == -1)
        {
            LOG_ERR(SYN_MEM_ALLOC, "Failed to unmap host memory on error {} {}", errno, strerror(errno));
        }
    }
    // no need to free "timestamp_elements"
}

synStatus ScalEventsPool::acquireNextAvailableTimestamp(TimestampBuff*& timestampBuff)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& eventTimeCollectionHandle : m_eventTimeCollectionHandles)
    {
        if (eventTimeCollectionHandle.second.unusedTimestamps != 0)
        {
            // find the index of the first free timestamp.
            unsigned firstAvailable = eventTimeCollectionHandle.second.unusedTimestamps._Find_first();
            // unset the 'firstAvailable' bit to 0 - to mark as taken
            eventTimeCollectionHandle.second.unusedTimestamps[firstAvailable] = 0;
            timestampBuff = &eventTimeCollectionHandle.second.timestampBuffers[firstAvailable];
            return synSuccess;
        }
    }
    // no free timestampBuffers - allocate now
    uint64_t cbOffsetInFd = 0;
    int      rc = hlthunk_allocate_timestamp_elements(m_fd, NUMBER_OF_TIME_COLLECTING_EVENTS_PER_HANDLE, &cbOffsetInFd);
    if (rc || !cbOffsetInFd)
    {
        LOG_ERR(SYN_STREAM, "{}: fd={} hlthunk_allocate_timestamp_elements error {})", HLLOG_FUNC, m_fd, rc);
        return synFail;
    }
    // map to host
    auto timestampsMappedMemory = (uint64_t*)mmap(nullptr,
                                                  NUMBER_OF_TIME_COLLECTING_EVENTS_PER_HANDLE * sizeof(uint64_t),
                                                  PROT_READ | PROT_WRITE,
                                                  MAP_SHARED,
                                                  m_fd,
                                                  cbOffsetInFd);
    if (timestampsMappedMemory == MAP_FAILED)
    {
        LOG_ERR(SYN_MEM_ALLOC, "Failed to map host memory on error {} {}", errno, strerror(errno));
    }
    m_eventTimeCollectionHandles[cbOffsetInFd].timestampsMappedMemory = timestampsMappedMemory;
    // initialize per event of the 64 events pool
    for (uint64_t timestampBufferIndex = 0; timestampBufferIndex < NUMBER_OF_TIME_COLLECTING_EVENTS_PER_HANDLE;
         timestampBufferIndex++)
    {
        m_eventTimeCollectionHandles[cbOffsetInFd].timestampBuffers[timestampBufferIndex] = {
            .cbOffsetInFd        = cbOffsetInFd,
            .indexInCbOffsetInFd = timestampBufferIndex,
            .timestamp           = timestampsMappedMemory + timestampBufferIndex};
        // set all as free except the first timestamp
        m_eventTimeCollectionHandles[cbOffsetInFd].unusedTimestamps.set();
        m_eventTimeCollectionHandles[cbOffsetInFd].unusedTimestamps[0] = 0;
    }

    timestampBuff = &m_eventTimeCollectionHandles[cbOffsetInFd].timestampBuffers[0];
    return synSuccess;
}
