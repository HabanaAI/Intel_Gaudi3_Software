#include "streams_container.hpp"
#include "stream.hpp"
#include "syn_logging.h"
#include "defs.h"
#include "defenders.h"
#include "global_statistics.hpp"

const uint64_t DEFAULT_STREAM_AFFINITY_MASK = 0x1;

StreamsContainer::StreamsContainer(bool setStreamAffinityByJobType)
: m_setStreamAffinityByJobType(setStreamAffinityByJobType), m_queueTypeUsageCounters {0, 0, 0, 0, 0}
{
}

synStatus StreamsContainer::createStream(synEventHandle   eventHandle,
                                         EventInterface&  rEventInterface,
                                         synStreamHandle* pStreamHandle)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_queueHandles.empty())
    {
        LOG_ERR(SYN_STREAM, "cannot create stream without queues");
        return synFail;
    }

    const unsigned streamAffinity = 0;
    auto result                   = m_streamHandles.insert(0, eventHandle, rEventInterface, streamAffinity, m_queueHandles[streamAffinity]);
    auto resultTyped              = std::make_pair((synStreamHandle)result.first, std::move(result.second));

    if (resultTyped.second != nullptr)
    {
        *pStreamHandle = resultTyped.first;
    }
    else
    {
        LOG_WARN(SYN_API, "createStream: Failed to create new stream");
        return synOutOfResources;
    }

    m_affinityUsageCounters[streamAffinity]++;

    return synSuccess;
}

synStatus StreamsContainer::destroyStream(synStreamHandle streamHandle)
{
    auto streamSptr = getStreamSptr(streamHandle);
    if (streamSptr)
    {
        synStatus status = streamSptr->synchronize();
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM,
                    "pStream {:#x} failed to streamSynchronize since synchronize failed with status {}",
                    TO64(streamSptr.get()),
                    status);
            return status;
        }
    }
    else
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed on stream handle {:x} verification, handle is incorrect, probably stream was already destroyed",
                __FUNCTION__,
                (SMHandle)streamHandle);
        return synInvalidArgument;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    const unsigned streamAffinity = streamSptr->getAffinity();
    m_affinityUsageCounters[streamAffinity]--;

    streamSptr.reset();
    synStatus ret = m_streamHandles.erase((SMHandle)streamHandle) ? synSuccess : synFail;
    if (ret != synSuccess)
    {
        LOG_ERR(SYN_API,
                "destroyStream: failed to destroy stream {}, maybe it is in use by another thread",
                (SMHandle)streamHandle);
    }

    return ret;
}

SlotMapItemSptr<Stream> StreamsContainer::getStreamSptr(synStreamHandle streamHandle) const
{
    return m_streamHandles[(SMHandle)streamHandle];
}

synStatus StreamsContainer::addStreamAffinities(const QueueInterfacesArrayVector& rQueueHandles)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    const unsigned sizeBefore = m_queueHandles.size();
    m_queueHandles.insert(m_queueHandles.end(),
                          std::make_move_iterator(rQueueHandles.begin()),
                          std::make_move_iterator(rQueueHandles.end()));
    const unsigned sizeAfter = m_queueHandles.size();

    // Fill nullptr handles with previous affinity handles
    if (sizeBefore > 0)
    {
        for (unsigned iter = sizeBefore; iter < sizeAfter; iter++)
        {
            const QueueInterfacesArray& rStreamHandlesBefore = m_queueHandles[iter - 1];
            QueueInterfacesArray&       rStreamHandles       = m_queueHandles[iter];

            for (unsigned typeIndex = QUEUE_TYPE_COPY_DEVICE_TO_HOST; typeIndex < QUEUE_TYPE_MAX_USER_TYPES;
                 typeIndex++)
            {
                if (rStreamHandles[typeIndex] == nullptr)
                {
                    rStreamHandles[typeIndex] = rStreamHandlesBefore[typeIndex];
                }
            }
        }
    }

    AffinityUsageVector affinityUsageCounter(sizeAfter - sizeBefore, 0);

    m_affinityUsageCounters.insert(m_affinityUsageCounters.end(),
                                   std::make_move_iterator(affinityUsageCounter.begin()),
                                   std::make_move_iterator(affinityUsageCounter.end()));

    return synSuccess;
}

synStatus StreamsContainer::removeAffinities(QueueInterfacesArrayVector& rQueueHandles)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_streamHandles.eraseAll();

    rQueueHandles.insert(rQueueHandles.end(),
                         std::make_move_iterator(m_queueHandles.begin()),
                         std::make_move_iterator(m_queueHandles.end()));
    m_queueHandles.clear();
    m_affinityUsageCounters.clear();
    std::fill(std::begin(m_queueTypeUsageCounters), std::end(m_queueTypeUsageCounters), 0);
    return synSuccess;
}

void StreamsContainer::getAffinities(uint64_t& rStreamAffinityMask) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    getAffinitiesUnlocked(rStreamAffinityMask);
}

void StreamsContainer::getAffinitiesUnlocked(uint64_t& rStreamAffinityMask) const
{
    rStreamAffinityMask = 0;

    for (unsigned iter = 0; iter < m_queueHandles.size(); iter++)
    {
        rStreamAffinityMask |= 1 << iter;
    }
}

synStatus StreamsContainer::addJob(Stream* pStream, std::unique_ptr<StreamJob>& rJob)
{
    CHECK_POINTER(SYN_STREAM, pStream, "pStream", synInvalidArgument);

    if (!pStream->isAffinityLocked())
    {
        STAT_GLBL_START(StreamLockAffinityDuration);
        std::lock_guard<std::mutex> lock(m_mutex);
        STAT_GLBL_COLLECT_TIME(StreamLockAffinityDuration, globalStatPointsEnum::StreamLockAffinityDuration);

        if (pStream->isAffinityLocked())
        {
            LOG_DEBUG_T(SYN_STREAM, "pStream {:#x} affinity is already set - skip", TO64(pStream));
        }
        else
        {
            synStatus status;
            if (m_setStreamAffinityByJobType)
            {
                if ((rJob->getType() != JobType::WAIT_FOR_EVENT) && (rJob->getType() != JobType::RECORD_EVENT))
                {
                    status = setStreamAffinityByJobType(pStream, rJob->getType());
                }
                else
                {
                    status = synSuccess;
                }
            }
            else
            {
                status = setStreamAffinityByMask(pStream, DEFAULT_STREAM_AFFINITY_MASK);
            }
            if (status != synSuccess)
            {
                LOG_ERR(SYN_STREAM, "pStream {:#x} failed to set affinity on addJob status {}", TO64(pStream), status);
                return status;
            }
        }
    }

    return pStream->addJob(rJob);
}

synStatus StreamsContainer::setStreamAffinity(Stream* pStream, uint64_t streamAffinityMask)
{
    STAT_GLBL_START(StreamLockAffinityDuration);
    std::lock_guard<std::mutex> lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(StreamLockAffinityDuration, globalStatPointsEnum::StreamLockAffinityDuration);

    CHECK_POINTER(SYN_STREAM, pStream, "pStream", synInvalidArgument);
    if (pStream->isAffinityLocked())
    {
        LOG_ERR(SYN_STREAM, "setStreamAffinity pStream {:#x} was called after affinity lock", TO64(pStream));
        return synFail;
    }

    return setStreamAffinityByMask(pStream, streamAffinityMask);
}

synStatus StreamsContainer::getStreamAffinity(Stream* pStream, uint64_t& rStreamAffinityMask) const
{
    std::unique_lock<std::mutex> lock(m_mutex);

    const unsigned streamAffinity = pStream->getAffinity();

    rStreamAffinityMask = 0;
    rStreamAffinityMask |= 1 << streamAffinity;
    return synSuccess;
}

synStatus StreamsContainer::getDeviceNextStreamAffinity(uint64_t& rDeviceAffinityMaskRange)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    uint64_t rStreamAffinityMask = 0;
    getAffinitiesUnlocked(rStreamAffinityMask);

    unsigned streamAffinityNew = getLruStreamAffinityByMask(rStreamAffinityMask);
    rDeviceAffinityMaskRange   = 1 << streamAffinityNew;
    return synSuccess;
}

synStatus StreamsContainer::synchronizeAllStreams()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    synStatus status = synSuccess;

    m_streamHandles.forEach([&status](Stream* pStream)
    {
        if (status == synSuccess)
        {
            status = Stream::synchronize(*pStream);
        }
    });

    return status;
}

void StreamsContainer::dump(synapse::LogManager::LogType logType) const
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_streamHandles.forEach(
        [&logType](const Stream* pStream) { SYN_LOG(logType, SPDLOG_LEVEL_INFO, "{}", pStream->toString()); });
}

synStatus StreamsContainer::setStreamAffinityByJobType(Stream* pStream, JobType jobType)
{
    QueueType queueType;
    synStatus status = getStreamType(jobType, queueType);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "pStream {:#x} getStreamType failed jobType {} status {}", TO64(pStream), jobType, status);
        return status;
    }

    const unsigned streamAffinityOld = pStream->getAffinity();
    const unsigned streamAffinityNew = getStreamAffinityByStreamType(queueType);

    status = pStream->setAffinity(streamAffinityNew, m_queueHandles[streamAffinityNew], jobType);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "pStream {:#x} setStreamAffinityByJobType failed jobType {} status {}",
                TO64(pStream),
                jobType,
                status);
        return status;
    }

    m_queueTypeUsageCounters[queueType]++;
    m_affinityUsageCounters[streamAffinityOld]--;
    m_affinityUsageCounters[streamAffinityNew]++;

    return synSuccess;
}

synStatus StreamsContainer::setStreamAffinityByMask(Stream* pStream, uint64_t streamAffinityMask)
{
    const unsigned streamAffinityOld = pStream->getAffinity();
    m_affinityUsageCounters[streamAffinityOld]--;
    const unsigned  streamAffinityNew = getLruStreamAffinityByMask(streamAffinityMask);
    const synStatus status            = pStream->setAffinity(streamAffinityNew, m_queueHandles[streamAffinityNew]);
    if (status != synSuccess)
    {
        m_affinityUsageCounters[streamAffinityOld]++;
        LOG_ERR(SYN_STREAM, "pStream {:#x} setStreamAffinity failed with status {}", TO64(pStream), status);
        return status;
    }

    m_affinityUsageCounters[streamAffinityNew]++;

    return synSuccess;
}

synStatus StreamsContainer::getStreamType(JobType jobType, QueueType& rStreamType)
{
    synStatus status;
    switch (jobType)
    {
        case JobType::MEMCOPY_H2D:
        {
            rStreamType = QUEUE_TYPE_COPY_HOST_TO_DEVICE;
            status      = synSuccess;
            break;
        }
        case JobType::MEMCOPY_D2H:
        case JobType::MEMSET:
        {
            rStreamType = QUEUE_TYPE_COPY_DEVICE_TO_HOST;
            status      = synSuccess;
            break;
        }
        case JobType::MEMCOPY_D2D:
        {
            rStreamType = QUEUE_TYPE_COPY_DEVICE_TO_DEVICE;
            status      = synSuccess;
            break;
        }
        case JobType::COMPUTE:
        {
            rStreamType = QUEUE_TYPE_COMPUTE;
            status      = synSuccess;
            break;
        }
        case JobType::NETWORK:
        {
            rStreamType = QUEUE_TYPE_NETWORK_COLLECTIVE;
            status      = synSuccess;
            break;
        }
        case JobType::RECORD_EVENT:
        case JobType::WAIT_FOR_EVENT:
        default:
        {
            status = synFail;
        }
    }
    return status;
}

unsigned StreamsContainer::getStreamAffinityByStreamType(QueueType queueType) const
{
    const unsigned affinityMax       = m_affinityUsageCounters.size();
    const unsigned lruStreamAffinity = m_queueTypeUsageCounters[queueType] % affinityMax;
    return lruStreamAffinity;
}

unsigned StreamsContainer::getLruStreamAffinityByMask(uint64_t streamAffinityMask) const
{
    uint64_t streamAffinityMaskLocal  = streamAffinityMask;
    unsigned lruStreamAffinity        = 0;
    unsigned lruStreamAffinityCounter = std::numeric_limits<unsigned>::max();

    for (unsigned index = 0; index < m_affinityUsageCounters.size(); index++)
    {
        if (streamAffinityMaskLocal & 0x1)
        {
            if (m_affinityUsageCounters[index] < lruStreamAffinityCounter)
            {
                lruStreamAffinity        = index;
                lruStreamAffinityCounter = m_affinityUsageCounters[index];
            }
        }
        else
        {
            // No need to iterate all 64 bits, if the mask is already 0 we can stop here
            if (streamAffinityMaskLocal == 0)
            {
                break;
            }
        }

        streamAffinityMaskLocal >>= 1;
    }

    return lruStreamAffinity;
}
