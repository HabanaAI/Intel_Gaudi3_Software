#pragma once
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <boost/interprocess/sync/posix/semaphore.hpp>
#include "synapse_common_types.h"
#include "synapse_api_types.h"
#include "runtime/common/common_types.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "hcl_public_streams.h"

struct tensor_info_t;
class StreamJob;
class QueueInterface;
class EventInterface;
class StreamsContainer;
enum class JobType;

class Stream
{
    friend class CounterStreamJobMock;

public:
    Stream(synEventHandle              eventHandle,
           EventInterface&             rEventInterface,
           unsigned                    streamAffinity,
           const QueueInterfacesArray& rStreamHandles);

    virtual ~Stream();

    inline synEventHandle getEventHandle() const { return m_eventHandle; }

    synStatus addJob(std::unique_ptr<StreamJob>& rJob);

    synStatus setAffinity(unsigned streamAffinity, const QueueInterfacesArray& rStreamHandles);

    synStatus setAffinity(unsigned streamAffinity, const QueueInterfacesArray& rStreamHandles, JobType jobType);

    inline unsigned getAffinity() const { return m_streamAffinity; }

    inline bool isAffinityLocked() const { return m_isAffinityLocked; }

    synStatus synchronize();

    static synStatus synchronize(Stream& stream) { return stream.synchronize(); };

    synStatus query();

    synStatus flushWaitsOnCollectiveQueue();

    uint32_t getPhysicalQueueOffset(QueueType queueType) const;

    hcl::hclStreamHandle getHclStreamHandle(QueueType queueType) const;

    synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                              std::vector<tensor_info_t>& tensorInfoArray) const;

    inline void testGetQueueInterface(QueueType queueType, QueueInterface*& rpQueueInterface) const
    {
        _getQueueInterface(queueType, rpQueueInterface);
    }

    std::string toString() const;

private:
    static std::string VirtualQueueTypeToStr(QueueType queueType);
    synStatus        _syncQueues(QueueInterface* queueInterfaceOld, QueueInterface* queueInterfaceNew);
    static QueueType _getQueueType(JobType jobType);
    inline void      _getQueueInterface(QueueType queueType, QueueInterface*& rpQueueInterface) const
    {
        _getQueueInterface(m_queueHandles, queueType, rpQueueInterface);
    }
    static void _getQueueInterface(const QueueInterfacesArray& rStreamHandles,
                                   QueueType                   queueType,
                                   QueueInterface*&            rpQueueInterface);
    void      _waitForEventQueueAdd(std::unique_ptr<StreamJob>& rJob);
    synStatus   _waitForEventQueueRun(QueueInterface* pQueueInterface);

    synStatus _addJobMultiThreads(std::unique_ptr<StreamJob>& rJob, QueueType queueType);

    synStatus _addJobSyncPhase(std::unique_ptr<StreamJob>& rJob,
                               QueueInterface**            ppQueueInterface,
                               QueueType                   queueType);
    synStatus _addJobExecutePhase(std::unique_ptr<StreamJob>& rJob, QueueInterface* pQueueInterface);

    const synEventHandle m_eventHandle;
    EventInterface&      m_rEventInterface;
    unsigned             m_streamAffinity;
    QueueInterfacesArray m_queueHandles;
    std::atomic<bool>    m_isAffinityLocked;
    QueueType            m_lastQueueType;
    std::atomic<bool>    m_isFirstJob;

    // The lax queue type mutex logic is as follows:
    // WR lock: change in m_lastQueueType
    // RD lock: otherwise
    mutable std::shared_mutex               m_lastQueueTypeMutex;

    // Locking a CS for threads for all queue-typem
    // until completion of jobs' execution by previous queue-type
    mutable std::mutex                      m_globalQueuesMutex;

    std::vector<std::unique_ptr<StreamJob>> m_waitForEventQueue;
    mutable std::mutex                      m_waitForEventQueueMutex;
};
