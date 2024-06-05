#pragma once

#include <deque>
#include <deque>
#include <mutex>
#include <unordered_map>
#include "synapse_types.h"
#include "synapse_common_types.hpp"
#include "synapse_api_types.h"
#include "internal/hccl_internal.h"

#include "infra/containers/slot_map_alloc.hpp"

#include "runtime/common/queues/event_with_mapped_tensor.hpp"

struct StreamIdToEventSequenceId
{
    uint64_t streamId;
    uint64_t eventSequenceId;
};

struct EventTimeReturn
{
    uint64_t seqId;
    uint64_t time;
};

class EventsPool;

/****************************************************************************************/
/****************************************************************************************/
/*                                                                                      */
/*                                   QmanEvent                                */
/*                                                                                      */
/****************************************************************************************/
/****************************************************************************************/
struct QmanEvent : public EventWithMappedTensor
{
    friend class lock;

public:
    QmanEvent(EventsPool* pool);

    QmanEvent() : QmanEvent(nullptr) {}  // Needed for the array of events

    void init(uint32_t eventIdx, unsigned int flags);
    void clear();

    synStatus synchronizeEvent() const;

    uint32_t getEventIdx() const { return m_eventIndex; };
    bool     getCollectTime() const { return m_collectTime; }
    uint64_t getHandle() const { return m_handle; }

    void setSignalSeqId(uint64_t streamId, uint64_t signalSeq) { m_streamIdToSignalSeq = {streamId, signalSeq}; };
    void clearDb() { m_streamIdToSignalSeq = {0, INVALID_SEQ_ID}; }

    uint64_t  getSeqIdsExcludeStreamId(uint64_t ignoreStreamId) const;
    synStatus eventHandleQueryNoLock(uint64_t timeout, EventTimeReturn* eventTime) const;
    synStatus eventHandleQuery(EventTimeReturn* eventTime = nullptr) const;

    void testingOnlySetSeqId(uint64_t seq) { m_streamIdToSignalSeq.eventSequenceId = seq; }

    std::string toString() const override;

private:
    synStatus getTime(uint64_t& nanoseconds, bool start) const override;

    void updateEventLock(bool isReadOperation) const;
    void updateEventUnlock(bool isReadOperation) const;

public:  // members from here
    // HCL requests (sadly) uses this specific handle
    hcclEventHandle handleRequest;

private:
    // A signal is a recorded status of a stream
    StreamIdToEventSequenceId m_streamIdToSignalSeq;

    // Todo - use the better SingleExecutionOwner mechanism
    bool             m_collectTime;
    mutable bool     m_isLocked;
    mutable uint32_t m_readCounter;

    union
    {
        uint32_t m_eventIndex;
        uint64_t m_handle;
    };
    EventsPool* m_pool;

public:
    class lock
    {
    public:
        lock(const QmanEvent* event, bool isReadOperation)
        : m_event(event), m_isReadOperation(isReadOperation)
        {
            m_event->updateEventLock(m_isReadOperation);
        }
        ~lock() { m_event->updateEventUnlock(m_isReadOperation); }

    private:
        const QmanEvent* m_event;
        bool             m_isReadOperation;
    };
};

/****************************************************************************************/
/****************************************************************************************/
/*                                                                                      */
/*                                      EventsPool                                      */
/*                                                                                      */
/****************************************************************************************/
/****************************************************************************************/
class EventsPool
{
public:
    EventsPool();
    ~EventsPool();

    bool                       getNewEvent(synEventHandle& newEventHandle, unsigned int flags);
    bool                       destroyEvent(synEventHandle eventHandle);
    SlotMapItemSptr<QmanEvent> getEventSptr(synEventHandle eventHandle);

    std::recursive_mutex m_mutex;

private:
    const uint32_t s_eventsInPool;

    ConcurrentSlotMapAlloc<QmanEvent> m_events;  // pool of all events
};
