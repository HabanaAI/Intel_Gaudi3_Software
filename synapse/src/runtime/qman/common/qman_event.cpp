#include "qman_event.hpp"
#include "syn_singleton.hpp"
#include "internal/hccl_internal.h"
#include "event_triggered_logger.hpp"
#include "habana_global_conf_runtime.h"

// In case we fail to lock an event we sleep and try again (TODO: write a reads/writers lock)
#define USEC_WAIT_FOR_LOCKED_EVENT_TIMEOUT 100

/****************************************************************************************/
/****************************************************************************************/
/*                                                                                      */
/*                                   QmanEvent                                */
/*                                                                                      */
/****************************************************************************************/
/****************************************************************************************/
/*
 ***************************************************************************************************
 *   @brief constructor for QmanEvent, pool is a pointer to the pool this event is in
 *
 *   @return None
 ***************************************************************************************************
 */
QmanEvent::QmanEvent(EventsPool* pool)
{
    m_isLocked            = false;
    m_collectTime         = false;
    m_readCounter         = 0;
    handleRequest         = hcclEventHandle();
    m_pool                = pool;  // points to the pool this event is in
    m_streamIdToSignalSeq = {0, INVALID_SEQ_ID};
}

/*
 ***************************************************************************************************
 *   @brief This function returns the time an event finished.
 *          It does some sanity checks, then calls event query to get a vector of all seqIds and time of each.
 *          It then goes over the vector, verifies all are valid (!=0)  and picks the min/max for start/end event
 *   @param nanoseconds - return value: when this event was finished.
 *          start       - Indicating if start/end event
 *   @return synStatus
 ***************************************************************************************************
 */
synStatus QmanEvent::getTime(uint64_t& nanoseconds, bool start) const
{
    std::string msg = start ? "Start" : "End";
    nanoseconds     = 0;

    EventTimeReturn seqAndTime;

    {
        QmanEvent::lock lock(this, true);
        // Verify time collection requested for start event
        if (!m_collectTime)
        {
            LOG_ERR_T(SYN_STREAM, "{}: {} Event is not collecting time {}", HLLOG_FUNC, msg.c_str(), toString());
            return synInvalidArgument;
        }

        if (m_streamIdToSignalSeq.eventSequenceId == INVALID_SEQ_ID)
        {
            LOG_WARN_T(SYN_STREAM,
                       "{}: {} Event {} has no seq id (was not recorded on any stream)",
                       HLLOG_FUNC,
                       msg.c_str(),
                       toString());
            return synObjectNotInitialized;
        }

        if (isInternalSignalingEvent())
        {
            LOG_ERR(SYN_STREAM,
                    "{}: not supported for external tensors mapped events, event tensor id {}",
                    HLLOG_FUNC,
                    getTensorIdx());
            return synInvalidArgument;
        }

        synStatus statusStart = eventHandleQueryNoLock(0, &seqAndTime);
        if (statusStart != synSuccess)
        {
            LOG_ERR_T(SYN_STREAM, "{}: Query on event {} returned status {}", HLLOG_FUNC, toString(), statusStart);
            return statusStart;
        }
    }
    uint64_t time = seqAndTime.time;
    if (time == 0)
    {
        LOG_WARN(SYN_STREAM, "{}: Could not get time for event {}, handle 0x{:x} - gone", HLLOG_FUNC, msg, m_handle);
        return synUnavailable;
    }
    // Find min or max time for event start or end (skip 0)

    nanoseconds = time;
    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief Queries an event (get status and end time). It is done by calling the driver (waitAndReleaseCS)
 *          The function assume the caller is holding the lock on the event
 *   @param eventTime - output. If the pointer is given, the functiion fills the time the event ended into it
 *
 *   @return synStatus
 ***************************************************************************************************
 */
synStatus QmanEvent::eventHandleQueryNoLock(uint64_t timeout, EventTimeReturn* eventTime) const
{
    synStatus status = synSuccess;
    if (eventTime != nullptr)
    {
        eventTime->seqId = INVALID_SEQ_ID;
        eventTime->time  = 0;
    }

    uint64_t seqId = m_streamIdToSignalSeq.eventSequenceId;
    if (seqId != INVALID_SEQ_ID)
    {
        if (seqId != 0)
        {
            uint64_t time = 0;
            status =
                _SYN_SINGLETON_INTERNAL->waitAndReleaseCS(seqId, SYNAPSE_WAIT_FOR_QUERY_TIMEOUT, true, false, &time);
            if (status != synSuccess)
            {
                LOG_DEBUG(SYN_STREAM,
                          "{}: Query event {} signal sequence-number 0x{:x} returned status {}",
                          HLLOG_FUNC,
                          toString(),
                          seqId,
                          status);
                return status;
            }
            if (eventTime != nullptr)
            {
                eventTime->seqId = seqId;
                eventTime->time  = time;
            }
        }
    }

#ifndef _POWER_PC_
    if (handleRequest.event != 0)
    {
        hcclResult_t hcclStatus = hcclSuccess;

        LOG_TRACE(SYN_STREAM, "{}: Found COLLECTIVE_NETWORK / SEND stream. Calling hcclSynchronizeEvent", HLLOG_FUNC);
        hcclStatus = hcclSynchronizeEvent(handleRequest, 0);

        switch (hcclStatus)
        {
            case hcclSuccess:
            case hcclInvalidArgument:
                break;

            case hcclBusy:
                status = synBusy;
                break;

            default:
                LOG_ERR(SYN_STREAM,
                        "{}: Calling hcclSynchronizeEvent failed. With event {}, hcclStatus={}",
                        HLLOG_FUNC,
                        handleRequest.event,
                        hcclStatus);
                break;
        }
    }
#endif

    return status;
}

/*
 ***************************************************************************************************
 *   @brief Queries an event: it locks the event, then calls eventHandleQueryNoLock and releases the lock
 *   @param eventTime - output. If the pointer is given, the functiion fills the time the event ended into it
 *
 *   @return synStatus
 ***************************************************************************************************
 */
synStatus QmanEvent::eventHandleQuery(EventTimeReturn* eventTime) const
{
    QmanEvent::lock lock(this, true);

    synStatus status = eventHandleQueryNoLock(SYNAPSE_WAIT_FOR_QUERY_TIMEOUT, eventTime);
    if (status != synSuccess)
    {
        LOG_DEBUG(SYN_STREAM, "{}: Query event {} main handle returned status {}", HLLOG_FUNC, toString(), status);
    }
    return status;
}

/*
 ***************************************************************************************************
 *   @brief Sets the data to the event (evnetIdx, flags)
 *   @param eventIdx, flags
 *   @return None
 ***************************************************************************************************
 */
void QmanEvent::init(uint32_t eventIdx, unsigned int flags)
{
    m_eventIndex  = eventIdx;
    m_collectTime = (bool)(flags & EVENT_COLLECT_TIME);
    clearDb();
}

/*
 ***************************************************************************************************
 *   @brief clears data of the event (HCL event, all seqIDs)
 *   @param None
 *   @return None
 ***************************************************************************************************
 */
void QmanEvent::clear()
{
    handleRequest = hcclEventHandle();
    clearDb();
}

/*
 ***************************************************************************************************
 *   @brief returns a vector of all the seqIds related to this event, that are not on the stream given by the user
 *   @param None
 *   @return Vector of seqIDs
 ***************************************************************************************************
 */
uint64_t QmanEvent::getSeqIdsExcludeStreamId(uint64_t ignoreStreamId) const
{
    if (m_streamIdToSignalSeq.streamId == ignoreStreamId)
    {
        return INVALID_SEQ_ID;
    }
    return m_streamIdToSignalSeq.eventSequenceId;
}

/*
 ***************************************************************************************************
 *   @brief Locks an event using readers/writers lock (note, it is using sleep!). It is using a shared mutex to make the
 *lock
 *   @param isReadOperation (read or write unlock)
 *   @return Vector of seqIDs
 ***************************************************************************************************
 */
void QmanEvent::updateEventLock(bool isReadOperation) const
{
    LOG_TRACE_T(SYN_STREAM, "Locking 0x{:x} read op {} readcounter 0x{:x}", TO64(this), isReadOperation, m_readCounter);
    // Ensure that this event will not be overriden by others
    // TODO - define when /  if it is user responsibility
    do
    {
        if ((!m_isLocked) && ((isReadOperation) || (m_readCounter == 0)))
        {
            std::unique_lock<std::recursive_mutex> lock(m_pool->m_mutex);
            if ((!m_isLocked) && ((isReadOperation) || (m_readCounter == 0)))
            {
                if (isReadOperation)
                {
                    m_readCounter++;
                }
                else
                {
                    m_isLocked = true;
                }
                break;
            }
        }
        // else - wait (a bit...)
        usleep(USEC_WAIT_FOR_LOCKED_EVENT_TIMEOUT);
    } while (1);
}

/*
 ***************************************************************************************************
 *   @brief Unlocks an event using readers/writers lock. It is using a shared mutex to make the unlock
 *   @param isrReadOperation (read or write unlock)
 *   @return Vector of seqIDs
 ***************************************************************************************************
 */
void QmanEvent::updateEventUnlock(bool isReadOperation) const
{
    LOG_TRACE_T(SYN_STREAM,
                "Unlocking 0x{:x} read op {} readcounter 0x{:x}",
                TO64(this),
                isReadOperation,
                m_readCounter);
    std::unique_lock<std::recursive_mutex> lock(m_pool->m_mutex);
    if (isReadOperation)
    {
        m_readCounter--;
    }
    else
    {
        m_isLocked = false;
    }
}

/*
 ***************************************************************************************************
 *   @brief Unlocks an event using readers/writers lock. It is using a shared mutex to make the unlock
 *   @param isrReadOperation (read or write unlock)
 *   @return Vector of seqIDs
 ***************************************************************************************************
 */
synStatus QmanEvent::synchronizeEvent() const
{
    uint64_t seqId = INVALID_SEQ_ID;
    hcclEventHandle handleRequestCopy;
    {
        // Ensure that this event will not be overriden by others
        QmanEvent::lock lock(this, true);
        seqId = m_streamIdToSignalSeq.eventSequenceId;
        handleRequestCopy = handleRequest;
    }
    if (seqId != INVALID_SEQ_ID && seqId != 0)
    {
        ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

        synStatus status = _SYN_SINGLETON_INTERNAL->waitAndReleaseCS(seqId, SYNAPSE_WAIT_FOR_CS_DEFAULT_TIMEOUT);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Failed to synchronize over event-index {} with sequence-number {}",
                    HLLOG_FUNC,
                    m_eventIndex,
                    seqId);
            ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CHECK_OPCODES);
            return status;
        }

        ETL_ADD_LOG_T_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                            logId,
                            SYN_STREAM,
                            "{}: Synchronized event-index {} with sequence-number {}",
                            HLLOG_FUNC,
                            m_eventIndex,
                            seqId);

        LOG_DEBUG(SYN_STREAM,
                  "{}: Synchronized event-index {} with sequence-number {}",
                  HLLOG_FUNC,
                  m_eventIndex,
                  seqId);
    }
    // We do NOT clear the sequence-ids that had been synchronized

#ifndef _POWER_PC_
    if (handleRequestCopy.event != 0)
    {
        hcclResult_t hcclStatus = hcclSuccess;
        LOG_TRACE(SYN_STREAM,
                  "{}: Waiting on event that triggered hcclEventRecord. Calling hcclSynchronizeEvent",
                  HLLOG_FUNC);
        hcclStatus = hcclSynchronizeEvent(handleRequestCopy, HCCL_InfinityWait);

        if (hcclStatus != hcclSuccess && hcclStatus != hcclInvalidArgument)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Calling hcclSynchronizeEvent failed. With event {}",
                    handleRequestCopy.event,
                    HLLOG_FUNC);
            return synFailHccl;
        }
    }
#endif

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief creates a string describing the event, used for logs
 *   @param None
 *   @return string
 ***************************************************************************************************
 */
std::string QmanEvent::toString() const
{
    return fmt::format("event 0x{:x} handle 0x{:x}", m_eventIndex, TO64(this));
}

/****************************************************************************************/
/****************************************************************************************/
/*                                                                                      */
/*                                      EventsPool                                      */
/*                                                                                      */
/****************************************************************************************/
/****************************************************************************************/
/*
 ***************************************************************************************************
 *   @brief Fills the array with default event constructor, puts all events (their pointers) into a free queue
 *   @param None
 *   @return None
 ***************************************************************************************************
 */
EventsPool::EventsPool()
: s_eventsInPool(GCFG_NUM_OF_USER_STREAM_EVENTS.value()), m_events(s_eventsInPool, SlotMapChecks::full, nullptr)
{
}

/*
 ***************************************************************************************************
 *   @brief Currently empty
 *   @param None
 *   @return None
 ***************************************************************************************************
 */
EventsPool::~EventsPool() {}

/*
 ***************************************************************************************************
 *   @brief gets a new event. Event is taken from the free events list and updated in the DB that holds the used events
 *and sets the needed value to the evetn
 *   @param rEventHandle - return event
 *          flags - data to be set in the event
 *   @return true (success), false (failed to find a free event)
 ***************************************************************************************************
 */
bool EventsPool::getNewEvent(synEventHandle& rEventHandle, unsigned int flags)
{
    auto newItem = m_events.insert(0, this);
    if (newItem.second != nullptr)
    {
        newItem.second->init(getSMHandleIndex(newItem.first), flags);
        rEventHandle = (synEventHandle)newItem.first;
        LOG_TRACE_T(SYN_STREAM, "{}: {}", HLLOG_FUNC, newItem.second->toString());
    }
    else
    {
        LOG_ERR_T(SYN_STREAM, "{}: No more free events for device", HLLOG_FUNC);
    }

    return newItem.second != nullptr;
}

/*
 ***************************************************************************************************
 *   @brief destroys an event, moving it from the used list back to the free list
 *   @param pEventHandle - event to be destroyed
 *
 *   @return true (success), false (failed to find the event the user gave)
 ***************************************************************************************************
 */
bool EventsPool::destroyEvent(synEventHandle eventHandle)
{
    auto eventSptr = getEventSptr(eventHandle);

    if (eventSptr == nullptr)
    {
        LOG_ERR_T(SYN_STREAM,
                  "{}: Can not destroy event, event handle {:x} not found",
                  HLLOG_FUNC,
                  (SMHandle)eventHandle);
        return false;
    }

    std::string eventDesc = eventSptr->toString();
    eventSptr.reset();
    if (!m_events.erase((SMHandle)eventHandle))
    {
        LOG_WARN_T(SYN_STREAM,
                   "{}: failed to destroy event handle: {:#x} desc: {}. maybe it was destroyed by another thread",
                   HLLOG_FUNC,
                   (SMHandle)eventHandle,
                   eventDesc);
        return false;
    }

    LOG_TRACE_T(SYN_STREAM, "{}: event handle: {:#x} desc: {}", HLLOG_FUNC, (SMHandle)eventHandle, eventDesc);

    return true;
}

SlotMapItemSptr<QmanEvent> EventsPool::getEventSptr(synEventHandle eventHandle)
{
    return m_events[(SMHandle)eventHandle];
}