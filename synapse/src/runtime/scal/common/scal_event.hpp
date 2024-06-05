#pragma once

#include "hcl_public_streams.h"

#include "infra/containers/slot_map_alloc.hpp"

#include "runtime/common/queues/event_with_mapped_tensor.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"

#include <unordered_map>
#include <mutex>
#include <bitset>

#include "syn_logging.h"

class QueueBaseScal;

struct TimestampBuff
{
    uint64_t  cbOffsetInFd;
    uint64_t  indexInCbOffsetInFd;
    volatile uint64_t* timestamp;
};
struct ScalEvent : public EventWithMappedTensor
{
public:
    ScalEvent(uint32_t devIdx) : ScalEvent(devIdx, 0, nullptr) {}
    ScalEvent(uint32_t devIdx, const unsigned int flags, TimestampBuff* timestampBuffer = nullptr);
    ScalEvent(const ScalEvent& other);

    void clearState() override;

    void setMappedTensor(uint64_t              tensorOffset,
                         uint32_t              tensorId,
                         const char*           tensorName,
                         InternalRecipeHandle* pInternalRecipeHandle) override;

    void setOnHclStream(bool onHclStream = true);

    bool        isOnHclStream() const { return m_isOnHclStream; }
    synStatus   getTime(uint64_t& nanoseconds, bool start) const override;
    std::string toString() const override
    {
        if (collectTime)
        {
            return fmt::format("handle 0x{:x} longSo (index {}, value {:#x}) {} time collecting on CB {} offset {} timestamp {}",
                            (uint64_t)(this),
                            longSo.m_index,
                            longSo.m_targetValue,
                            isOnHclStream() ? std::string("on hcl stream") : std::string(),
                            timestampBuff->cbOffsetInFd,
                            timestampBuff->indexInCbOffsetInFd,
                            *timestampBuff->timestamp);
        }
        else
        {
            return fmt::format("handle 0x{:x} longSo (index {}, value {:#x}) {}",
                            (uint64_t)(this),
                            longSo.m_index,
                            longSo.m_targetValue,
                            isOnHclStream() ? std::string("on hcl stream") : std::string());
        }
    }

    ScalLongSyncObject longSo        = {};
    QueueBaseScal*     pStreamIfScal = nullptr;
    hcl::syncInfo      hclSyncInfo   = {};
    synEventHandle     eventHandle   = nullptr;
    bool               collectTime   = false;
    TimestampBuff*     timestampBuff = nullptr;

    void lock() const;
    void unlock() const;

private:
    bool m_isOnHclStream = false;
    mutable std::mutex m_mutex;
};

class ScalEventsPool
{
public:
    ScalEventsPool(int fd);
    virtual ~ScalEventsPool();

    std::pair<synEventHandle, SlotMapItemSptr<ScalEvent>> getNewEvent(unsigned int flags);
    SlotMapItemSptr<ScalEvent>                            getEventSptr(synEventHandle eventHandle);
    synStatus                                             destroyEvent(synEventHandle eventHandle);

private:
    synStatus                               acquireNextAvailableTimestamp(TimestampBuff*& timestampBuff);
    static constexpr uint32_t               NUMBER_OF_TIME_COLLECTING_EVENTS_PER_HANDLE = 64;
    int                                     m_fd                                        = 0;
    ConcurrentSlotMapAlloc<ScalEvent, 2048> m_eventsDB;
    std::mutex                              m_mutex;

    struct TimestampsHandle
    {
        // each bit represent a timestamp in the 64 timestamps batch - 0 is used, 1 is free.
        std::bitset<NUMBER_OF_TIME_COLLECTING_EVENTS_PER_HANDLE> unusedTimestamps;
        TimestampBuff timestampBuffers[NUMBER_OF_TIME_COLLECTING_EVENTS_PER_HANDLE];
        uint64_t*     timestampsMappedMemory;
    };
    std::unordered_map<uint64_t /*eventTimeCollectionCbOffsetInFd*/, TimestampsHandle> m_eventTimeCollectionHandles;
};
