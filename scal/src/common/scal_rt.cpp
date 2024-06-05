#include <assert.h>
#include <cstring>
#include <string>
#include "scal.h"
#include "scal_utilities.h"
#include "scal_base.h"
#include "logger.h"
#include "hlthunk.h"


int Scal::streamSetBuffer(StreamInterface *stream, Buffer *buffer)
{
    if (stream == nullptr)
    {
        LOG_ERR(SCAL,"{}, Invalid stream", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return stream->setBuffer(buffer);
}

int Scal::streamSetPriority(StreamInterface *stream, unsigned priority)
{
    if (stream == nullptr)
    {
        LOG_ERR(SCAL,"{}, Invalid stream", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    return stream->setPriority(priority);
}

int Scal::getHostFenceCounterInfo(const HostFenceCounter * hostFenceCounter, scal_host_fence_counter_info_t *info)
{
    if (hostFenceCounter->completionGroup == nullptr)
    {
        LOG_ERR(SCAL, "{}: Host fence counter has no completion group", __FUNCTION__);
        return SCAL_NOT_FOUND;
    }

    info->name                 = hostFenceCounter->name.c_str();
    info->sm                   = hostFenceCounter->syncManager->smIndex;
    info->cq_index             = hostFenceCounter->completionGroup->cqIdx;
    info->so_index             = hostFenceCounter->soIdx;
    info->master_monitor_index = hostFenceCounter->monBase;
    info->ctr                  = hostFenceCounter->completionGroup->pCounter;
    info->request_counter      = &hostFenceCounter->requestCounter;
    info->isr_enabled          = hostFenceCounter->isrEnable;

    return SCAL_SUCCESS;
}

int Scal::hostFenceCounterWait(const HostFenceCounter *hostFenceCounter, const uint64_t num_credits, uint64_t timeout)
{
    if (hostFenceCounter->completionGroup == nullptr)
    {
        LOG_ERR(SCAL, "{}: Host fence counter has no completion group", __FUNCTION__);
        return SCAL_NOT_FOUND;
    }

    auto target = num_credits + hostFenceCounter->requestCounter;
    hostFenceCounter->requestCounter = target;
    Scal* scal = (Scal*)hostFenceCounter->scal;
    return scal->completionGroupWait(hostFenceCounter->completionGroup, target, timeout, false);
}

int Scal::hostFenceCounterEnableIsr(HostFenceCounter *hostFenceCounter, bool enableIsr)
{
    if (hostFenceCounter->completionGroup == nullptr)
    {
        LOG_ERR(SCAL, "{}: Host fence counter has no completion group", __FUNCTION__);
        return SCAL_NOT_FOUND;
    }
    Scal *scal = hostFenceCounter->scal;
    assert(scal != nullptr);
    if (scal == nullptr)
    {
        LOG_ERR(SCAL, "{}: Host fence counter has no scal set", __FUNCTION__);
        return SCAL_NOT_FOUND;
    }
    if (enableIsr == hostFenceCounter->isrEnable)
    {
        return SCAL_SUCCESS;
    }

    hostFenceCounter->completionGroup->isrIdx = enableIsr ? hostFenceCounter->isrIdx : scal_illegal_index;
    hostFenceCounter->isrEnable = enableIsr;

    scal->enableHostFenceCounterIsr(hostFenceCounter->completionGroup, enableIsr);
    return SCAL_SUCCESS;
}

/**
 * local macro for logging in completionGroupWait
 * allows choosing logging level
 * @param logLevel logging logLevel (such as spdlog::level::trace)
 * @param cg pointer to completion group
 * @param target target we're waiting for
 */
#define LOG_CG_BY_LEVEL(logLevel, cg, target) do { \
            HLLOG_BY_LEVEL_F(SCAL, logLevel, "{}", cg->getLogInfo(target)); \
    } while (0)

int Scal::completionGroupWait(const CompletionGroupInterface *completionGroup, const uint64_t target, uint64_t timeout, bool alwaysWaitForInterrupt)
{
    // waits on the fence counter to reach a specific target value.
    // interrupt_idx : when configuring the cq:  256 + dcore_id*64 ( 64 per dcore) + cq_id.
    // wait on LKD until it returns
    // implemented by waiting on isrIdx.
    uint32_t status = HL_WAIT_CS_STATUS_COMPLETED;
    if (!alwaysWaitForInterrupt)
    {
        // check if completion group has already reached the target to avoid call to ioctl in hlthunk_wait_for_interrupt
        if (*(completionGroup->pCounter) >= target)
        {
            return SCAL_SUCCESS;
        }
        if (timeout == 0)
        {
            return SCAL_TIMED_OUT;
        }
    }

    uint64_t counterArray[c_max_cq_cntrs];

    const Scal* scal = completionGroup->scal;
    uint32_t ctrArraySize = scal->getCqsSize();
    if (ctrArraySize > c_max_cq_cntrs)
    {
        LOG_ERR(SCAL, "{}: ctrArraySize={} exceeds {} (update c_max_cq_cntrs)",
                __FUNCTION__, ctrArraySize, c_max_cq_cntrs);
        return SCAL_FAILURE;
    }

    bool scalForever = (timeout == SCAL_FOREVER);
    if (scalForever)
    {
        timeout = scal->m_timeoutUsNoProgress * 2; // set the time to user-request*2 so TDR will catch timeouts and
                                                   // not this logic.Keep this one for cases that the TDR doesn't
                                                   // catch (HCL for example)
        LOG_TRACE(SCAL, "{}: timeout SCAL_FOREVER was converted locally to {} microseconds", __FUNCTION__, timeout);
    }

    bool timeoutDisabled = scal->m_timeoutDisabled;
    bool progress        = false; // if any completion group counter changed since last loop
    bool firstNoProgress = true; // flag for printing once when there's no progress
    // the loop iterates only if SCAL_FOREVER received. Otherwise, it finishes after one iteration.
    // if timeout is disabled, the loop runs until completion, or until an error.
    // otherwise, the loop runs until completion, until an error, or until there's no progress on any completion group.
    for (uint64_t loopCtr = 1; true; loopCtr++)
    {
        LOG_DEBUG(SCAL, "{}: hlthunk_wait_for_interrupt_by_handle() addr={:#x} value {:#x} irq {:#x} timeout={:#x}", __FUNCTION__,
                  *(completionGroup->pCounter), target, completionGroup->isrIdx, timeout);
        LOG_CG_BY_LEVEL(HLLOG_LEVEL_DEBUG, completionGroup, target);
        int rc = 0;
        if (completionGroup->isrIdx == scal_illegal_index)
        {
            auto start = std::chrono::steady_clock::now();
            uint64_t timeUs = 0;
            while(true)
            {
                // busy wait.
                // use internal loop to minimize latency and not get current time after each counter reading
                for (unsigned i = 0 ; i < 100; ++i)
                {
                    if (*(completionGroup->pCounter) >= target)
                    {
                        return SCAL_SUCCESS;
                    }
                }
                if (timeUs >= timeout)
                {
                    status = HL_WAIT_CS_STATUS_BUSY;
                    break;
                }
                timeUs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
            }
        }
        else
        {
            rc = hlthunk_wait_for_interrupt_by_handle(scal->getFD(),
                                                      completionGroup->scal->m_completionQueuesHandle,
                                                      completionGroup->globalCqIndex,
                                                      target,
                                                      completionGroup->isrIdx,
                                                      timeout,
                                                      &status);
        }

        if (!rc && status == HL_WAIT_CS_STATUS_COMPLETED)   // completed successfully
        {
            LOG_INFO(SCAL, "{}: hlthunk_wait_for_interrupt_by_handle completed successfully", __FUNCTION__);
            return SCAL_SUCCESS;
        }

        if (rc && errno != ETIMEDOUT)       // error which isn't timeout
        {
            LOG_ERR(SCAL, "{}: hlthunk_wait_for_interrupt_by_handle failed. rc={} status={} errno={} error: '{}'",
                    __FUNCTION__, rc, status, errno, std::strerror(errno));
            LOG_CG_BY_LEVEL(HLLOG_LEVEL_ERROR, completionGroup, target);
            return SCAL_FAILURE;
        }

        if (status == HL_WAIT_CS_STATUS_BUSY || (rc && errno == ETIMEDOUT))     // waiting timed out
        {
            if (!scalForever)   // will finish after one iteration
            {
                LOG_DEBUG(SCAL, "{}: hlthunk_wait_for_interrupt_by_handle is still busy after {} microseconds", __FUNCTION__, timeout);
                LOG_CG_BY_LEVEL(HLLOG_LEVEL_DEBUG, completionGroup, target);
                return SCAL_TIMED_OUT;
            }

            LOG_WARN(SCAL, "{}: hlthunk_wait_for_interrupt_by_handle is busy for the {} time", __FUNCTION__, loopCtr);

            // init counters only in the first loop
            if (loopCtr == 1)
            {
                initCountersArray(counterArray, ctrArraySize);
                continue;
            }

            // update the counters array and check for progress (some counter increment)
            updateCountersArray(counterArray, ctrArraySize, progress);
            if (!progress)
            {
                LOG_ERR(SCAL, "{}: waiting failed. No progress on any completion group after timeout of {} microseconds", __FUNCTION__, timeout);
                if (firstNoProgress)
                {
                    firstNoProgress = false;
                    LOG_CG_BY_LEVEL(HLLOG_LEVEL_ERROR, completionGroup, target);
                    logCompletionGroupsCtrs();
                }
                if (!timeoutDisabled)       // when timeout isn't disabled, return. If timeout IS disabled, we continue looping
                {
                    return SCAL_TIMED_OUT;
                }
            }
        }
        else        // didn't complete and didn't time out => failed
        {
            LOG_ERR(SCAL, "{}: hlthunk_wait_for_interrupt_by_handle failed with status {}", __FUNCTION__, status);
            LOG_CG_BY_LEVEL(HLLOG_LEVEL_ERROR, completionGroup, target);
            return SCAL_FAILURE;
        }
    }
}

#undef LOG_CG_BY_LEVEL

/**
 * logs current values of all completion group counters
 */
void Scal::logCompletionGroupsCtrs() const
{
    std::basic_string<char> completionGroupName;

    uint64_t counter = 0;
    for (auto cgEntry : m_cgs)
    {
        // completionGroupName = cgEntry.first;
        counter = *(cgEntry->pCounter);
        LOG_TRACE(SCAL, "{}: PDMA direct Completion Group='{}' counter={}", __FUNCTION__, cgEntry->name, counter);
    }
}

/**
 * fills an array with the completion groups counters
 * @param countersArray an array in the size of arraySize, for holding completion queues counters
 * @param arraySize size of the given array (should be equal to size of m_completionGroups)
 */
void Scal::initCountersArray(uint64_t countersArray[], unsigned long arraySize) const
{
    int index = 0;
    for (auto& completionGroup : m_cgs)
    {
        countersArray[index] = *(completionGroup->pCounter);
        index++;
    }
    LOG_TRACE(SCAL, "{}: counters array was initialized", __FUNCTION__);
}

/**
 * receives an array of completion groups counters, updates the current counters and sets 'true' value in "inProgress"
 * if some counter was incremented comparing to the previous value in the array
 * @param countersArray an array in the size of arraySize, for holding completion queues counters
 * @param arraySize size of the given array (should be equal to size of m_completionGroups)
 * @param inProgress will be equal true if some completion counter in the array had progress
 */
void Scal::updateCountersArray(uint64_t countersArray[], unsigned long arraySize, bool &inProgress) const
{
    uint64_t prevCounter, currentCounter;
    inProgress = false;
    int index = 0;

    // iterate linearly over the completion groups and array and fill the counter values. If any counter was increased, set inProgress to be true.
    for (auto& completionGroup : m_cgs)
    {
        prevCounter = countersArray[index];
        currentCounter = *(completionGroup->pCounter);
        if (currentCounter != prevCounter)
        {
            LOG_TRACE(SCAL, "{}: completionGroup='{}' counter incremented from {} to {}",
                        __FUNCTION__, completionGroup->name, prevCounter, currentCounter);

            inProgress = true;
        }
        countersArray[index] = currentCounter;
        index++;
    }

    if (!inProgress)
    {
        LOG_WARN(SCAL, "{}: counters array was updated, but no counters were changed", __FUNCTION__);
    }
}

int Scal::completionGroupRegisterTimestamp(const CompletionGroupInterface* completionGroup,
                                           const uint64_t                  target,
                                           const uint64_t                  timestampsHandle,
                                           const uint32_t                  timestampsOffset)
{
    int      fd               = 0;
    unsigned isrIdx           = 0;
    uint64_t cqCountersHandle = 0;
    unsigned cqIndex          = 0;
    completionGroup->getTimestampInfo(fd, isrIdx, cqCountersHandle, cqIndex);

    int rc = hlthunk_register_timestamp_interrupt(fd, isrIdx, cqCountersHandle, cqIndex,
                                                  target, timestampsHandle, timestampsOffset);
    if (rc)
    {
        LOG_ERR(SCAL, "{}: hlthunk_register_timestamp_interrupt CQ {} IDX {} ISR {} timestampsHandle {:#x} timestampsOffset {} - failed errno {} {} ", __FUNCTION__,
                completionGroup->name, completionGroup->globalCqIndex, completionGroup->isrIdx, timestampsHandle, timestampsOffset,
                errno, std::strerror(errno));
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

const Scal::Pool * Scal::getPoolByName(const std::string & poolName) const
{
    if (m_pools.find(poolName) != m_pools.end())
    {
        return &m_pools.at(poolName);
    }
    return nullptr;
}

const Scal::Pool * Scal::getPoolByID(const unsigned poolID) const
{
    for (auto & pool : m_pools)
    {
        if (pool.second.globalIdx == poolID)
        {
            return &pool.second;
        }
    }
    return nullptr;
}

Scal::Core* Scal::getCoreByName(const std::string& coreName) const
{
    for (auto & core : m_cores)
    {
        if (core && (core->name == coreName))
        {
            return core;
        }
    }
    return nullptr;
}

const Scal::Core * Scal::getCoreByID(const unsigned coreID) const
{
    if (coreID < m_cores.size())
    {
        return m_cores[coreID];
    }
    return nullptr;
}

const Scal::StreamInterface* Scal::getStreamByName(const std::string &streamName) const
{
    if (m_streams.find(streamName) != m_streams.end())
    {
        return &m_streams.at(streamName);
    }
    else if (m_directModePdmaChannelStreams.find(streamName) != m_directModePdmaChannelStreams.end())
    {
        return m_directModePdmaChannelStreams.at(streamName);
    }
    return nullptr;
}

const Scal::StreamSet *Scal::getStreamSetByName(const std::string &streamSetName) const
{
    if (m_streamSets.find(streamSetName) != m_streamSets.end())
    {
        return &m_streamSets.at(streamSetName);
    }
    return nullptr;
}

const Scal::Stream * Scal::getStreamByID(const Scheduler * scheduler, const unsigned streamID) const
{
    if (scheduler)
    {
        if (streamID < scheduler->streams.size())
        {
            return scheduler->streams[streamID];
        }
    }
    return nullptr;
}

const Scal::CompletionGroupInterface * Scal::getCompletionGroupByName(const std::string & completionGroupName) const
{
    if (m_completionGroups.find(completionGroupName) != m_completionGroups.end())
    {
        return &m_completionGroups.at(completionGroupName);
    }
    else if (m_directModeCompletionGroups.find(completionGroupName) != m_directModeCompletionGroups.end())
    {
        return m_directModeCompletionGroups.at(completionGroupName);
    }

    return nullptr;
}

const Scal::Cluster* Scal::getClusterByName(const std::string  &clusterName) const
{
    if (m_clusters.find(clusterName) != m_clusters.end())
    {
        return &m_clusters.at(clusterName);
    }
    return nullptr;
}

const Scal::SyncObjectsPool *Scal::getSoPool(const std::string &poolName) const
{
    //TBD
    if(m_soPools.find(poolName) != m_soPools.end())
    {
        return &m_soPools.at(poolName);

    }
    return nullptr;
}

const Scal::SyncManager * Scal::getSyncManager(const unsigned index) const
{
    if (index < m_syncManagers.size())
    {
        return &m_syncManagers[index];
    }
    return nullptr;
}

const Scal::MonitorsPool *Scal::getMonitorPool(const std::string &poolName) const
{
    if( m_monitorPools.find(poolName) != m_monitorPools.end())
    {
        return &m_monitorPools.at(poolName);

    }
    return nullptr;
}

const Scal::HostFenceCounter *Scal::getHostFenceCounter(const std::string &counterName) const
{
    if (m_hostFenceCounters.find(counterName) != m_hostFenceCounters.end())
    {
        return &m_hostFenceCounters.at(counterName);
    }
    return nullptr;
}

unsigned  Scal::getFWCombinedVersion(bool scheduler) const
{
    if (scheduler)
        return m_fw_sched_major_version * 0x10000 + m_fw_sched_minor_version;
    else
        return m_fw_eng_major_version * 0x10000 + m_fw_eng_minor_version;
}

int Scal::bgWork(void (*logFunc)(int, const char*), char *errMsg, int errMsgSize)
{
    if (!m_bgWork)
    {
        // We might get here in case we releaseDevice but scal inialization wasn't fully completed
        return SCAL_SUCCESS;
    }

    return m_bgWork->tdr(logFunc, errMsg, errMsgSize);
}

int Scal::debugBackgroundWork()
{
    if (!m_bgWork)
    {
        // We might get here in case we releaseDevice but scal inialization wasn't fully completed
        return SCAL_SUCCESS;
    }

    return m_bgWork->debugCheckStatus();
}

int Scal::runCores(const uint32_t* coreIds, const uint32_t* coreQmanIds, uint32_t numOfCores)
{
    if (numOfCores == 0)
    {
        return SCAL_SUCCESS;
    }

    if ((coreIds == nullptr) || (coreQmanIds == nullptr))
    {
        LOG_ERR(SCAL,"{}: nullptr paramter coreIds {} coreQmanIds {}",
                __FUNCTION__, (uint64_t) coreIds, (uint64_t) coreQmanIds);
        assert(0);
        return SCAL_FAILURE;
    }

    // request run cores
    int rc = hlthunk_engines_command(m_fd, coreIds, numOfCores, HL_ENGINE_CORE_RUN);
    if (rc != 0)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to run cores (amount {}) rc {}", __FUNCTION__, m_fd, numOfCores, rc);
        assert(0);
        return SCAL_FAILURE;
    }

    Qman::Workload workload;

    const uint32_t* currCoreQmanId = coreQmanIds;
    for (uint32_t i = 0; i < numOfCores; i++, currCoreQmanId++)
    {
        Qman::Program program;

        addFencePacket(program, 0 /* id */, 1 /* targetVal */, 1 /* decVal */);
        workload.addProgram(program, *currCoreQmanId);
    }

    if (!submitQmanWkld(workload))
    {
        LOG_ERR(SCAL,"{}: fd={} workload submit failed", __FUNCTION__, m_fd);
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}
