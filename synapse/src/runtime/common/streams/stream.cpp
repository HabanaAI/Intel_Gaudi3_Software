#include "stream.hpp"
#include "queues/queue_interface.hpp"
#include "synapse_common_types.h"
#include "stream_job.hpp"
#include "global_statistics.hpp"
#include "defs.h"
#include "defenders.h"

Stream::Stream(synEventHandle              eventHandle,
               EventInterface&             rEventInterface,
               unsigned                    streamAffinity,
               const QueueInterfacesArray& rQueueHandles)
: m_eventHandle(eventHandle),
  m_rEventInterface(rEventInterface),
  m_streamAffinity(streamAffinity),
  m_queueHandles(rQueueHandles),
  m_isAffinityLocked(false),
  m_lastQueueType(QUEUE_TYPE_COPY_DEVICE_TO_HOST),
  m_isFirstJob(true)
{
    LOG_INFO_T(SYN_STREAM,
               "Stream: {:#x} created eventHandle {:#x} m_streamAffinity {} "
               "dmaUpQueueHandle {:#x} dmaDownQueueandle {:#x} dmaD2dQueueHandle {:#x} "
               "computeQueueHandle {:#x} collectiveQueueHandle {:#x} m_lastQueueType {}",
               TO64(this),
               TO64(eventHandle),
               m_streamAffinity,
               TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_HOST]),
               TO64(m_queueHandles[QUEUE_TYPE_COPY_HOST_TO_DEVICE]),
               TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_DEVICE]),
               TO64(m_queueHandles[QUEUE_TYPE_COMPUTE]),
               TO64(m_queueHandles[QUEUE_TYPE_NETWORK_COLLECTIVE]),
               TO64(m_lastQueueType));
}

Stream::~Stream()
{
    LOG_INFO_T(SYN_STREAM, "Stream: {:#x} destroyed", TO64(this));
}

std::string Stream::VirtualQueueTypeToStr(QueueType queueType)
{
    switch (queueType)
    {
        case QUEUE_TYPE_COPY_DEVICE_TO_HOST:
            return "DMA_UP";
        case QUEUE_TYPE_COPY_HOST_TO_DEVICE:
            return "DMA_DOWN";
        case QUEUE_TYPE_COPY_DEVICE_TO_DEVICE:
            return "DMA_D2D";
        case QUEUE_TYPE_COMPUTE:
            return "COMPUTE";
        case QUEUE_TYPE_NETWORK_COLLECTIVE:
            return "NETWORK";
        default:
            return "";
    };
}

void Stream::_getQueueInterface(const QueueInterfacesArray& rQueueInterfaces,
                                QueueType                   queueType,
                                QueueInterface*&            rpQueueInterface)
{
    HB_ASSERT(queueType < QUEUE_TYPE_MAX_USER_TYPES, "Stream invalid call detected queueType {}", queueType);
    QueueInterface* pQueueInterface = rQueueInterfaces[queueType];
    HB_ASSERT(pQueueInterface != nullptr, "Stream invalid call detected queueType {} null queue", queueType);
    rpQueueInterface = pQueueInterface;
}

synStatus Stream::_syncQueues(QueueInterface* queueInterfaceOld, QueueInterface* queueInterfaceNew)
{
    synStatus status = queueInterfaceOld->eventRecord(m_rEventInterface, (synStreamHandle)this);
    if (status != synSuccess)
    {
        return status;
    }

    status = queueInterfaceNew->eventWait(m_rEventInterface, 0, (synStreamHandle)this);
    if (status != synSuccess)
    {
        return status;
    }

    return synSuccess;
}

synStatus Stream::addJob(std::unique_ptr<StreamJob>& rJob)
{
    STAT_GLBL_START(StreamAddJob);

    LOG_INFO_T(SYN_STREAM, "Stream {:#x} addJob was called with job {}", TO64(this), rJob->getDescription());

    switch (rJob->getType())
    {
        case JobType::MEMCOPY_H2D:
        case JobType::MEMCOPY_D2H:
        case JobType::MEMCOPY_D2D:
        case JobType::MEMSET:
        case JobType::COMPUTE:
        case JobType::NETWORK:
        {
            if (!m_isAffinityLocked)
            {
                LOG_ERR(SYN_STREAM,
                        "Stream {:#x} failed to addJob {} since affinity is not locked",
                        TO64(this),
                        rJob->getDescription());
                return synFail;
            }

            QueueType queueType = _getQueueType(rJob->getType());

            // Multi-threads execution mode
            synStatus status    = _addJobMultiThreads(rJob, queueType);
            if (status != synSuccess)
            {
                return status;
            }

            break;
        }
        case JobType::RECORD_EVENT:
        {
            // Note, record event will be executed (run) in parallel unless m_lastQueueType is about to be changed.
            std::shared_lock lock(m_lastQueueTypeMutex);

            m_isFirstJob = false; // we run the waits queue

            const QueueType queueType = m_lastQueueType;
            QueueInterface* pQueueInterface;
            _getQueueInterface(queueType, pQueueInterface);

            synStatus status = _waitForEventQueueRun(pQueueInterface);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_STREAM,
                        "Stream {:#x} failed to addJob {} since _waitForEventQueueRun failed with status",
                        TO64(this),
                        rJob->getDescription(),
                        status);
                return status;
            }

            status = rJob->run(pQueueInterface);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_STREAM,
                        "Stream {:#x} failed to addJob {} since run failed with status",
                        TO64(this),
                        rJob->getDescription(),
                        status);
                return status;
            }
            break;
        }
        case JobType::WAIT_FOR_EVENT:
        {
            _waitForEventQueueAdd(rJob);
            break;
        }
        default:
        {
            HB_ASSERT(false, "Invalid job type {}", rJob->getDescription());
        }
    }

    STAT_GLBL_COLLECT_TIME(StreamAddJob, globalStatPointsEnum::StreamAddJob);
    return synSuccess;
}

QueueType Stream::_getQueueType(JobType jobType)
{
    QueueType queueType;
    switch (jobType)
    {
        case JobType::MEMCOPY_D2H:
        {
            queueType = QUEUE_TYPE_COPY_DEVICE_TO_HOST;
            break;
        }
        case JobType::MEMCOPY_H2D:
        {
            queueType = QUEUE_TYPE_COPY_HOST_TO_DEVICE;
            break;
        }
        case JobType::MEMCOPY_D2D:
        case JobType::MEMSET:
        {
            queueType = QUEUE_TYPE_COPY_DEVICE_TO_DEVICE;
            break;
        }
        case JobType::COMPUTE:
        {
            queueType = QUEUE_TYPE_COMPUTE;
            break;
        }
        case JobType::NETWORK:
        {
            queueType = QUEUE_TYPE_NETWORK_COLLECTIVE;
            break;
        }
        case JobType::RECORD_EVENT:
        case JobType::WAIT_FOR_EVENT:
        default:
        {
            HB_ASSERT(false, "Invalid job type {}", TO64(jobType));
            queueType = QUEUE_TYPE_MAX_USER_TYPES;
        }
    }
    return queueType;
}

synStatus Stream::setAffinity(unsigned streamAffinity, const QueueInterfacesArray& rQueueHandles)
{
    LOG_INFO_T(SYN_STREAM, "Stream {:#x} setStreamAffinity was called, streamAffinity {}", TO64(this), streamAffinity);

    if (m_isAffinityLocked)
    {
        LOG_ERR(SYN_STREAM, "Stream {:#x} setStreamAffinity failed since affinity is already locked", TO64(this));
        return synFail;
    }

    m_queueHandles     = rQueueHandles;
    m_streamAffinity   = streamAffinity;
    m_isAffinityLocked = true;

    LOG_DEBUG_T(SYN_STREAM,
                "Stream: {:#x} called setStreamAffinity m_streamAffinity {} m_isAffinityLocked {} "
                "dmaUpQueueHandle {:#x} dmaDownQueueHandle {:#x} dmaD2dQueueHandle {:#x} computeQueueHandle {:#x}"
                " collectiveNetworkHandle {:#x}",
                TO64(this),
                m_streamAffinity,
                m_isAffinityLocked,
                TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_HOST]),
                TO64(m_queueHandles[QUEUE_TYPE_COPY_HOST_TO_DEVICE]),
                TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_DEVICE]),
                TO64(m_queueHandles[QUEUE_TYPE_COMPUTE]),
                TO64(m_queueHandles[QUEUE_TYPE_NETWORK_COLLECTIVE]));

    return synSuccess;
}

synStatus Stream::setAffinity(unsigned streamAffinity, const QueueInterfacesArray& rQueueHandles, JobType jobType)
{
    LOG_INFO_T(SYN_STREAM, "Stream {:#x} setStreamAffinity was called, streamAffinity {}", TO64(this), streamAffinity);

    if (m_isAffinityLocked)
    {
        LOG_ERR(SYN_STREAM, "Stream {:#x} setStreamAffinity failed since affinity is already locked", TO64(this));
        return synFail;
    }

    if (m_streamAffinity != streamAffinity)
    {
        const QueueType queueType = _getQueueType(jobType);
        if (!m_isFirstJob)
        {
            QueueInterface* pQueueInterfaceOld;
            _getQueueInterface(m_lastQueueType, pQueueInterfaceOld);

            QueueInterface* pQueueInterfaceNew;
            _getQueueInterface(rQueueHandles, queueType, pQueueInterfaceNew);

            LOG_INFO_T(SYN_STREAM,
                       "Stream {:#x} sync different affinities m_streamAffinity {} streamAffinity {} queues "
                       "queueHandleOld {:#x} queueHandleNew {:#x}",
                       TO64(this),
                       m_streamAffinity,
                       streamAffinity,
                       TO64(pQueueInterfaceOld),
                       TO64(pQueueInterfaceNew));

            synStatus status = _syncQueues(pQueueInterfaceOld, pQueueInterfaceNew);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_STREAM, "Stream {:#x} failed since _syncQueues failed with status", TO64(this), status);
                return status;
            }
        }

        m_lastQueueType = queueType;
    }

    m_queueHandles     = rQueueHandles;
    m_streamAffinity   = streamAffinity;
    m_isAffinityLocked = true;

    LOG_DEBUG_T(SYN_STREAM,
                "Stream: {:#x} called setStreamAffinity m_streamAffinity {} m_isAffinityLocked {} "
                "dmaUpQueueHandle {:#x} dmaDownQueueHandle {:#x} dmaD2dQueueHandle {:#x} computeQueueHandle {:#x}"
                " collectiveNetworkHandle {:#x}",
                TO64(this),
                m_streamAffinity,
                m_isAffinityLocked,
                TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_HOST]),
                TO64(m_queueHandles[QUEUE_TYPE_COPY_HOST_TO_DEVICE]),
                TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_DEVICE]),
                TO64(m_queueHandles[QUEUE_TYPE_COMPUTE]),
                TO64(m_queueHandles[QUEUE_TYPE_NETWORK_COLLECTIVE]));

    return synSuccess;
}

synStatus Stream::synchronize()
{
    STAT_GLBL_START(StreamSynchronize);

    std::shared_lock lock(m_lastQueueTypeMutex);

    m_isFirstJob = false; // we run the waits queue
    LOG_INFO_T(SYN_STREAM, "Stream {:#x} Synchronizing m_lastQueueType {}", TO64(this), m_lastQueueType);

    QueueInterface* pQueueInterface;
    _getQueueInterface(m_lastQueueType, pQueueInterface);

    synStatus status = _waitForEventQueueRun(pQueueInterface);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "Stream {:#x} failed to synchronize since _waitForEventQueueRun failed with status {}",
                TO64(this),
                status);
        return status;
    }

    status = pQueueInterface->synchronize((synStreamHandle)this, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "Stream {:#x} failed to synchronize since synchronize failed with status {}",
                TO64(this),
                status);
        return status;
    }

    STAT_GLBL_COLLECT_TIME(StreamSynchronize, globalStatPointsEnum::StreamSynchronize);
    return synSuccess;
}

synStatus Stream::query()
{
    STAT_GLBL_START(StreamQuery);

    std::shared_lock lock(m_lastQueueTypeMutex);

    m_isFirstJob = false; // we run the waits queue
    LOG_INFO_T(SYN_STREAM, "Stream {:#x} Query m_lastQueueType {}", TO64(this), m_lastQueueType);

    QueueInterface* pQueueInterface;
    _getQueueInterface(m_lastQueueType, pQueueInterface);

    synStatus status = _waitForEventQueueRun(pQueueInterface);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "Stream {:#x} failed to query since _waitForEventQueueRun failed with status {}",
                TO64(this),
                status);
        return status;
    }

    status = pQueueInterface->query();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "Stream {:#x} failed to query since query failed with status {}", TO64(this), status);
        return status;
    }

    STAT_GLBL_COLLECT_TIME(StreamQuery, globalStatPointsEnum::StreamQuery);
    return synSuccess;
}

void Stream::_waitForEventQueueAdd(std::unique_ptr<StreamJob>& rJob)
{
    STAT_GLBL_START(StreamWaitDuration);
    std::lock_guard<std::mutex> lock(m_waitForEventQueueMutex);
    STAT_GLBL_COLLECT_TIME(StreamWaitDuration, globalStatPointsEnum::StreamLockEventWaitDuration);
    m_waitForEventQueue.push_back(std::move(rJob));
}

synStatus Stream::_waitForEventQueueRun(QueueInterface* pQueueInterface)
{
    STAT_GLBL_START(StreamWaitDuration);
    std::lock_guard<std::mutex> lock(m_waitForEventQueueMutex);
    STAT_GLBL_COLLECT_TIME(StreamWaitDuration, globalStatPointsEnum::StreamLockEventWaitDuration);

    LOG_TRACE_T(SYN_STREAM,
                "Stream {:#x} _waitForEventQueueRun called with waitListSize {}",
                TO64(this),
                m_waitForEventQueue.size());

    if (!m_waitForEventQueue.empty())
    {
        for (auto jobIter = m_waitForEventQueue.begin(); jobIter != m_waitForEventQueue.end();)
        {
            synStatus status = (*jobIter)->run(pQueueInterface);
            if (status != synSuccess)
            {
                return status;
            }
            jobIter = m_waitForEventQueue.erase(jobIter);
        }
    }
    return synSuccess;
}

synStatus Stream::flushWaitsOnCollectiveQueue()
{
    LOG_TRACE_T(SYN_STREAM, "Stream {:#x} flushWaitOnCollectiveQueue called", TO64(this));
    QueueInterface* pQueueInterface;
    _getQueueInterface(QUEUE_TYPE_NETWORK_COLLECTIVE, pQueueInterface);
    return _waitForEventQueueRun(pQueueInterface);
}

uint32_t Stream::getPhysicalQueueOffset(QueueType queueType) const
{
    CHECK_POINTER(SYN_STREAM, m_queueHandles[queueType], "m_queueHandles[queueType]", 0);
    return m_queueHandles[queueType]->getPhysicalQueueOffset();
}

hcl::hclStreamHandle Stream::getHclStreamHandle(QueueType queueType) const
{
    CHECK_POINTER(SYN_STREAM, m_queueHandles[queueType], "m_queueHandles[queueType]", nullptr);
    return m_queueHandles[queueType]->getHclStreamHandle();
}

synStatus Stream::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                  std::vector<tensor_info_t>& tensorInfoArray) const
{
    CHECK_POINTER(SYN_STREAM,
                  m_queueHandles[QUEUE_TYPE_COMPUTE],
                  "m_queueHandles[QUEUE_TYPE_COMPUTE]",
                  synInvalidArgument);
    return m_queueHandles[QUEUE_TYPE_COMPUTE]->getDynamicShapesTensorInfoArray(recipeHandle, tensorInfoArray);
}

std::string Stream::toString() const
{
    return fmt::format("Stream: {:#x} eventHandle {:#x} m_streamAffinity {} "
                       "dmaUpQueueHandle {:#x} dmaDownQueueandle {:#x} dmaD2dQueueHandle {:#x} "
                       "computeQueueHandle {:#x} collectiveQueueHandle {:#x} m_lastQueueType {} m_isFirstJob {} "
                       "m_waitForEventQueue.size {}",
                       TO64(this),
                       TO64(m_eventHandle),
                       m_streamAffinity,
                       TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_HOST]),
                       TO64(m_queueHandles[QUEUE_TYPE_COPY_HOST_TO_DEVICE]),
                       TO64(m_queueHandles[QUEUE_TYPE_COPY_DEVICE_TO_DEVICE]),
                       TO64(m_queueHandles[QUEUE_TYPE_COMPUTE]),
                       TO64(m_queueHandles[QUEUE_TYPE_NETWORK_COLLECTIVE]),
                       TO64(m_lastQueueType),
                       TO64(m_isFirstJob),
                       m_waitForEventQueue.size());
}

synStatus Stream::_addJobMultiThreads(std::unique_ptr<StreamJob>& rJob, QueueType queueType)
{
    // lockWaiting blocks same queue types execution - it is going to be released later on to allow parallelism
    STAT_GLBL_START(StreamLockExistingJobWaitingDuration);
    std::unique_lock<std::mutex> lockWaiting(m_globalQueuesMutex);
    STAT_GLBL_COLLECT_TIME(StreamLockExistingJobWaitingDuration,
                      globalStatPointsEnum::StreamLockExistingJobWaitingDuration);

    QueueInterface* pQueueInterface(nullptr);
    synStatus       status = _addJobSyncPhase(rJob, &pQueueInterface, queueType);
    if (status != synSuccess)
    {
        return status;
    }

    status = _addJobExecutePhase(rJob, pQueueInterface);

    return status;
}

synStatus Stream::_addJobSyncPhase(std::unique_ptr<StreamJob>& rJob,
                                   QueueInterface**            ppQueueInterface,
                                   QueueType                   queueType)
{
    HB_ASSERT(ppQueueInterface != nullptr, "ppQueueInterface is nullptr");

    QueueInterface*& pQueueInterface = *ppQueueInterface;
    _getQueueInterface(queueType, pQueueInterface);

    synStatus status = _waitForEventQueueRun(pQueueInterface);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "Stream {:#x} failed to addJob {} since _waitForEventQueueRun failed with status",
                TO64(this),
                rJob->getDescription(),
                status);
        return status;
    }

    if (queueType != m_lastQueueType)
    {
        std::unique_lock lastQueueTypeMutex(m_lastQueueTypeMutex);

        if (!m_isFirstJob)
        {
            HB_ASSERT(m_lastQueueType != QUEUE_TYPE_MAX_USER_TYPES,
                        "Stream {:#x} Invalid call detected queueType {}",
                        TO64(this),
                        (unsigned)m_lastQueueType);

            LOG_INFO_T(SYN_STREAM,
                        "Stream {:#x} syncQueue was called, sync {} -> {}",
                        TO64(this),
                        VirtualQueueTypeToStr(m_lastQueueType),
                        VirtualQueueTypeToStr(queueType));

            QueueInterface* queueInterfaceOld;
            _getQueueInterface(m_lastQueueType, queueInterfaceOld);

            QueueInterface* queueInterfaceNew;
            _getQueueInterface(queueType, queueInterfaceNew);

            status = _syncQueues(queueInterfaceOld, queueInterfaceNew);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_STREAM,
                        "Stream {:#x} failed to addJob {} since _syncQueues failed with status",
                        TO64(this),
                        rJob->getDescription(),
                        status);
                return status;
            }
        }
        m_lastQueueType = queueType;
    }

    m_isFirstJob = false;

    return synSuccess;
}

synStatus Stream::_addJobExecutePhase(std::unique_ptr<StreamJob>& rJob, QueueInterface* pQueueInterface)
{
    // Note, in case the job type is known, its jobs will be executed (run) in parallel
    synStatus status = rJob->run(pQueueInterface);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "Stream {:#x} failed to addJob {} since run failed with status",
                TO64(this),
                rJob->getDescription(),
                status);
        return status;
    }

    return synSuccess;
}