#pragma once

#include <map>
#include <unordered_set>
#include <mutex>
#include "runtime/common/common_types.hpp"
#include "stream_job.hpp"
#include "infra/containers/slot_map_alloc.hpp"

class EventInterface;
class Stream;

class StreamsContainer
{
public:
    StreamsContainer(bool setStreamAffinityByJobType);

    ~StreamsContainer() = default;

    synStatus
    createStream(synEventHandle eventHandle, EventInterface& rEventInterface, synStreamHandle* rStreamHandle);

    synStatus destroyStream(synStreamHandle streamHandle);

    SlotMapItemSptr<Stream> getStreamSptr(synStreamHandle streamHandle) const;

    synStatus addStreamAffinities(const QueueInterfacesArrayVector& rQueueHandles);

    synStatus removeAffinities(QueueInterfacesArrayVector& rQueueHandles);

    void getAffinities(uint64_t& rStreamAffinityMask) const;

    synStatus addJob(Stream* pStream, std::unique_ptr<StreamJob>& rJob);

    synStatus setStreamAffinity(Stream* pStream, uint64_t streamAffinityMask);

    synStatus getStreamAffinity(Stream* pStream, uint64_t& rStreamAffinityMask) const;

    synStatus getDeviceNextStreamAffinity(uint64_t& rDeviceAffinityMaskRange);

    synStatus synchronizeAllStreams();

    void dump(synapse::LogManager::LogType logType) const;

private:
    static synStatus getStreamType(JobType jobType, QueueType& rStreamType);

    unsigned getStreamAffinityByStreamType(QueueType queueType) const;

    unsigned getLruStreamAffinityByMask(uint64_t streamAffinityMask) const;

    synStatus setStreamAffinityByJobType(Stream* pStream, JobType jobType);

    synStatus setStreamAffinityByMask(Stream* pStream, uint64_t streamAffinityMask);

    void getAffinitiesUnlocked(uint64_t& rStreamAffinityMask) const;

    using AffinityUsageVector  = std::vector<unsigned>;
    using QueueTypesUsageArray = std::array<unsigned, QUEUE_TYPE_MAX_USER_TYPES>;

    const bool                             m_setStreamAffinityByJobType;
    mutable std::mutex                     m_mutex;
    mutable ConcurrentSlotMapAlloc<Stream> m_streamHandles;
    QueueInterfacesArrayVector             m_queueHandles;
    AffinityUsageVector                    m_affinityUsageCounters;
    QueueTypesUsageArray                   m_queueTypeUsageCounters;
};
