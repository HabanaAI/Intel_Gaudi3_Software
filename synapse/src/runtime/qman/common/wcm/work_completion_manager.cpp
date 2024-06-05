#include "work_completion_manager.hpp"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "habana_global_conf_runtime.h"
#include "profiler_api.hpp"
#include "wcm_observer_interface.hpp"
#include "wcm_cs_querier_interface.hpp"
#include "global_statistics.hpp"
#include "hlthunk.h"
#include "debug_define.hpp"
#include "event_triggered_logger.hpp"

//WCM multi CS query duration at LKD in microseconds units.
//Driver returns earlier in case one of CSs completed. 0 duration means non-blocking query
//The LKD measures duration in Jiffies: 1/CONFIG_HZ (measured in SW-61258 as 4ms).
#define WCM_CS_QUERY_TIMEOUT_MULTI 5000000;

WorkCompletionManager::WorkCompletionManager(WcmCsQuerierInterface* pQuerier)
: m_pQuerier(pQuerier),
  m_stat("WorkCompletionManager ", 0, true /*enable*/),
  m_reportAmount(GCFG_WCM_REPORT_AMOUNT.value()),
  m_thread(nullptr),
  m_operation(0),
  m_preProcessedIncompleteCsHandlesStats {},
  m_postProcessedCompleteCsHandlesStats {}
{
    // Todo avoid using:
    // - QUEUE_ID_SIZE_MAX and work according to the given device types queues ID max
    // - use handle between the WCM and its I/Fs
    LOG_DEBUG(SYN_WORK_COMPL, "{}", HLLOG_FUNC);
}

WorkCompletionManager::~WorkCompletionManager()
{
    if (m_thread != nullptr)
    {
        stop();
    }
}

void WorkCompletionManager::start()
{
    if (m_thread == nullptr)
    {
        m_thread = new std::thread(&WorkCompletionManager::mainLoop, this);
        HB_ASSERT(m_thread != nullptr, "{}: Failure during WCM thread execution", __FUNCTION__);
        LOG_TRACE(SYN_WORK_COMPL, "{} WCM thread start", HLLOG_FUNC);
    }
}

void WorkCompletionManager::stop()
{
    LOG_TRACE(SYN_WORK_COMPL, "{}", HLLOG_FUNC);
    if (m_thread != nullptr)
    {
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_operation |= OPERATION_FINISH;
        }
        m_cv.notify_one();
        m_thread->join();
        LOG_TRACE(SYN_WORK_COMPL, "WorkCompletionManagerSingle::thread finished!");
        delete m_thread;
        m_thread = nullptr;
    }
}

void WorkCompletionManager::addCs(WcmPhysicalQueuesId phyQueId, WcmObserverInterface* pObserver, WcmCsHandle csHandle)
{
    STAT_GLBL_START(wcmAddDuration);
    LOG_TRACE(SYN_WORK_COMPL,
              "{} phyQueId {} pListener 0x{:x} csHandle {}",
              HLLOG_FUNC,
              phyQueId,
              TO64(pObserver),
              csHandle);

    {
        // Todo SW-62562 consider removing WCM multi observer locking on addCs
        std::unique_lock<std::mutex> lk(m_mutex);
        // Todo reduce the duration of addCs by pre-allocating memory (can't be done using the existing deque)
        storeCsHandle(m_incompleteCsHandles, phyQueId, {pObserver, csHandle, TimeTools::timeNow()});
        m_operation |= OPERATION_WORK;
    }
    m_cv.notify_one();
    STAT_GLBL_COLLECT_TIME(wcmAddDuration, globalStatPointsEnum::wcmAddDuration);
}

void WorkCompletionManager::storeCsHandle(PhysicalQueuesIdListenerCsQueueArr& rCsHandles,
                                          WcmPhysicalQueuesId                 phyQueId,
                                          const ListenerCs&                   rListenerCs)
{
    rCsHandles[phyQueId].push_back(rListenerCs);
}

void WorkCompletionManager::storeCsHandlesQueue(PhysicalQueuesIdListenerCsQueueArr& rCsHandles,
                                                WcmPhysicalQueuesId                 phyQueId,
                                                ListenerCsQueue&                    rListenerCsQueue,
                                                bool                                isFront)
{
    ListenerCsQueue&                rPhyListenerCsQueue = rCsHandles[phyQueId];
    ListenerCsQueue::const_iterator iter {(isFront || rPhyListenerCsQueue.empty()) ? rPhyListenerCsQueue.cbegin()
                                                                                   : rPhyListenerCsQueue.cend() - 1};
    rPhyListenerCsQueue.insert(iter,
                               std::make_move_iterator(rListenerCsQueue.begin()),
                               std::make_move_iterator(rListenerCsQueue.end()));
    rListenerCsQueue.clear();
}

void WorkCompletionManager::storeCsHandlesArr(PhysicalQueuesIdListenerCsQueueArr& rQueryIncompleteCsHandles,
                                              const PhysicalQueuesIdArray&        rPhyQueues,
                                              const ObserverArray&                rObservers,
                                              const CsHandleArray&                rCsHandles,
                                              const TimeArray&                    rTimes,
                                              uint32_t                            amount)
{
    for (uint64_t handlesAmountHandled = 0; handlesAmountHandled < amount; handlesAmountHandled++)
    {
        storeCsHandle(
            rQueryIncompleteCsHandles,
            rPhyQueues[handlesAmountHandled],
            {rObservers[handlesAmountHandled], rCsHandles[handlesAmountHandled], rTimes[handlesAmountHandled]});
    }
}

void WorkCompletionManager::dump()
{
    LOG_TRACE(SYN_WORK_COMPL, "{}", HLLOG_FUNC);

    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_operation |= OPERATION_DUMP;
    }
    m_cv.notify_one();
}

void WorkCompletionManager::mainLoop()
{
    while (true)
    {
        bool                               work;
        PhysicalQueuesIdListenerCsQueueArr csIncomplete;
        bool                               dump;
        bool                               finish;

        // WCM thread waits for {new CS , dump , finish} permutation event
        waitForEvent(work, csIncomplete, dump, finish);

        // Todo delete this statistics (added only to intercept unexpected executions)
        if (!work && !finish && !dump)
        {
            m_stat.collect(StatPoints::wcmNoWork, 1);
        }

        if (work)
        {
            const synStatus status = processCsHandles(csIncomplete);
            if (status == synDeviceReset)
            {
                LOG_CRITICAL(SYN_WORK_COMPL,
                             "{} WCM thread stop running since the LKD is expected to reset the device",
                             HLLOG_FUNC);
                finish = true;
            }
        }

        if (dump)
        {
            processDumpStatistics();
        }

        if (finish)
        {
            return;
        }
    }
}

void WorkCompletionManager::waitForEvent(bool&                               rWork,
                                         PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles,
                                         bool&                               rDump,
                                         bool&                               rFinish)
{
    std::unique_lock<std::mutex> lk(m_mutex);
    m_cv.wait(lk, [&] {
        rWork = (m_operation & OPERATION_WORK) == OPERATION_WORK;
        if (rWork)
        {
            swap(m_incompleteCsHandles, rIncompleteCsHandles);
        }
        rDump       = (m_operation & OPERATION_DUMP) == OPERATION_DUMP;
        rFinish     = (m_operation & OPERATION_FINISH) == OPERATION_FINISH;
        m_operation = 0;
        return (rWork || rFinish || rDump);
    });
}

void WorkCompletionManager::getWork(bool& rWork, PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles)
{
    std::unique_lock<std::mutex> lk(m_mutex);
    rWork = (m_operation & OPERATION_WORK) == OPERATION_WORK;

    if (rWork)
    {
        swap(m_incompleteCsHandles, rIncompleteCsHandles);
        m_operation &= ~OPERATION_WORK;
    }
}

synStatus WorkCompletionManager::processCsHandles(PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles)
{
    storePreProcessedIncompleteCsHandles(rIncompleteCsHandles);
    PhysicalQueuesIdListenerCsQueueArr completedCsHandles;

    while (true)
    {
        PhysicalQueuesIdArray phyQueues {0};
        ObserverArray         observers {0};
        CsHandleArray         csHandles {0};
        TimeArray             times {};
        uint32_t              amountQuery = 0;

        initializeCsHandles(rIncompleteCsHandles, phyQueues, observers, csHandles, times, amountQuery);

        if (amountQuery < csHandles.size())
        {
            bool                               work;
            PhysicalQueuesIdListenerCsQueueArr incompleteCsHandlesNew;
            getWork(work, incompleteCsHandlesNew);

            if (work)
            {
                storePreProcessedIncompleteCsHandles(incompleteCsHandlesNew);
                storePhysicalArr(incompleteCsHandlesNew, rIncompleteCsHandles, false);
                initializeCsHandles(rIncompleteCsHandles, phyQueues, observers, csHandles, times, amountQuery);
            }

            if (amountQuery == 0)
            {
                break;
            }
        }

        PhysicalQueuesIdListenerCsQueueArr queryIncompleteCsHandles;
        PhysicalQueuesIdListenerCsQueueArr queryCompleteCsHandles;

        uint32_t        amountCompleted;
        uint64_t        wcmTimeout = WCM_CS_QUERY_TIMEOUT_MULTI;
        const synStatus status = queryCsHandles(phyQueues,
                                                observers,
                                                csHandles,
                                                times,
                                                amountQuery,
                                                wcmTimeout,
                                                queryIncompleteCsHandles,
                                                queryCompleteCsHandles,
                                                amountCompleted);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_WORK_COMPL, "{} queryCsHandles failed with status {}", HLLOG_FUNC, status);
            dumpCsHandles(true,
                          queryIncompleteCsHandles,
                          queryCompleteCsHandles,
                          rIncompleteCsHandles,
                          completedCsHandles);

            if (status == synDeviceReset)
            {
                LOG_CRITICAL(SYN_WORK_COMPL, "{} device reset detected. Aborting query", HLLOG_FUNC);
                handleCompleteCsHandles(queryCompleteCsHandles, completedCsHandles);
                handleRejectedCsHandles(queryIncompleteCsHandles, rIncompleteCsHandles);
                return status;
            }

            LOG_WARN(SYN_WORK_COMPL, "{} WCM sleeping in case of LKD error {} ms ...", HLLOG_FUNC, wcmTimeout);
            std::this_thread::sleep_for(std::chrono::microseconds(wcmTimeout));
            LOG_WARN(SYN_WORK_COMPL, "{} WCM sleeping in case of LKD error {} ms Done", HLLOG_FUNC, wcmTimeout);
        }
        else if (amountCompleted == 0)
        {
            LOG_WARN(SYN_WORK_COMPL, "{} queryCsHandles returned without any CS completion", HLLOG_FUNC);
            dumpCsHandles(false,
                          queryIncompleteCsHandles,
                          queryCompleteCsHandles,
                          rIncompleteCsHandles,
                          completedCsHandles);
        }

        // queryCompleteCsHandles is empty after this call
        recycleCsHandles(queryCompleteCsHandles, false, completedCsHandles);

        // queryIncompleteCsHandles is empty after this call
        handleInflightCsHandles(queryIncompleteCsHandles, rIncompleteCsHandles);
    }

    storePostProcessedCompleteCsHandles(completedCsHandles);

    queryCsHandlesReport();

    return synSuccess;
}

void WorkCompletionManager::queryCsHandlesReport()
{
    reportPreProcessedStatistics();
    reportPostProcessedStatistics();
    m_pQuerier->report();
}

synStatus WorkCompletionManager::queryCsHandles(PhysicalQueuesIdArray&              rPhyQueues,
                                                ObserverArray&                      rObservers,
                                                CsHandleArray&                      rCsHandles,
                                                TimeArray&                          rTimes,
                                                uint32_t                            amountQuery,
                                                uint64_t                            timeoutUs,
                                                PhysicalQueuesIdListenerCsQueueArr& rQueryIncompleteCsHandles,
                                                PhysicalQueuesIdListenerCsQueueArr& rQueryCompleteCsHandles,
                                                uint32_t&                           rAmountCompleted)
{
    m_stat.collect(StatPoints::wcmQuery, 1);
    m_stat.collect(StatPoints::wcmQueryCsAmount, amountQuery);

    hlthunk_wait_multi_cs_in  inParams {rCsHandles.data(), timeoutUs, amountQuery};
    hlthunk_wait_multi_cs_out outParams {0};

    STAT_GLBL_START(wcmQueryDurationFail);
    STAT_GLBL_START(wcmQueryDurationTimeout);
    STAT_GLBL_START(wcmQueryDurationComplete);
    const synStatus status = m_pQuerier->query(&inParams, &outParams);
    LOG_TRACE(SYN_WORK_COMPL, "{}: status {} {}/{} completed", HLLOG_FUNC, status, outParams.completed, amountQuery);

    if (status != synSuccess)
    {
        STAT_GLBL_COLLECT_TIME(wcmQueryDurationFail, globalStatPointsEnum::wcmQueryDurationFail);
        m_stat.collect(StatPoints::wcmQueryFail, 1);
        LOG_ERR(SYN_WORK_COMPL, "{} amountQuery {} status {} query failed", HLLOG_FUNC, amountQuery, status);
        storeCsHandlesArr(rQueryIncompleteCsHandles, rPhyQueues, rObservers, rCsHandles, rTimes, amountQuery);
        return status;
    }

    rAmountCompleted = 0;
    if (outParams.completed == 0)
    {
        STAT_GLBL_COLLECT_TIME(wcmQueryDurationTimeout, globalStatPointsEnum::wcmQueryDurationTimeout);
        m_stat.collect(StatPoints::wcmQueryTimeout, 1);
        LOG_DEBUG(SYN_WORK_COMPL,
                  "{} amountQuery {} query timeout seq_set 0b{:b}",
                  HLLOG_FUNC,
                  amountQuery,
                  outParams.seq_set);
        storeCsHandlesArr(rQueryIncompleteCsHandles, rPhyQueues, rObservers, rCsHandles, rTimes, amountQuery);
        return synSuccess;
    }

    STAT_GLBL_COLLECT_TIME(wcmQueryDurationComplete, globalStatPointsEnum::wcmQueryDurationComplete);
    m_stat.collect(StatPoints::wcmQueryComplete, outParams.completed);
    LOG_TRACE(SYN_WORK_COMPL,
              "{} amountQuery {} query success completed {} seq_set 0b{:b}",
              HLLOG_FUNC,
              amountQuery,
              outParams.completed,
              outParams.seq_set);

    for (uint64_t currentPosIndex = 0, currentPosMask = 0x1; currentPosIndex < amountQuery;
         currentPosIndex++, currentPosMask <<= 1)
    {
        if (outParams.seq_set & currentPosMask)
        {
            storeCsHandle(rQueryCompleteCsHandles,
                          rPhyQueues[currentPosIndex],
                          {rObservers[currentPosIndex], rCsHandles[currentPosIndex], rTimes[currentPosIndex]});
            rAmountCompleted++;
        }
        else
        {
            storeCsHandle(rQueryIncompleteCsHandles,
                          rPhyQueues[currentPosIndex],
                          {rObservers[currentPosIndex], rCsHandles[currentPosIndex], rTimes[currentPosIndex]});
        }
    }

    HB_ASSERT(outParams.completed == rAmountCompleted,
              "{}: outParams.completed {} outParams.seq_set 0b{:b} rAmountCompleted {}",
              __FUNCTION__,
              outParams.completed,
              outParams.seq_set,
              rAmountCompleted);

    return synSuccess;
}

void WorkCompletionManager::processDumpStatistics() const
{
    dumpPreProcessedStatistics();
    dumpPostProcessedStatistics();
}

void WorkCompletionManager::dumpStatistics(WcmPhysicalQueuesId phyQueId, const CsStat& rStat)
{
    const DurationNs durationAvg = rStat.mDurationTotal / rStat.mAmount;
    LOG_DEBUG(SYN_WORK_COMPL,
              "phyQueId 0x{:x} amount {} total duration (ns) {} max duration (ns) {} avg duration (ns) {}",
              phyQueId,
              rStat.mAmount,
              rStat.mDurationTotal,
              rStat.mDurationMax,
              durationAvg);
}

void WorkCompletionManager::dumpStatistics(const StatArray& rStats)
{
    for (unsigned phyQueId = 0; phyQueId < rStats.size(); phyQueId++)
    {
        if (rStats[phyQueId].mAmount > 0)
        {
            dumpStatistics(phyQueId, rStats[phyQueId]);
        }
    }
}

void WorkCompletionManager::reportStatistics(uint64_t reportAmount, StatArray& rStats)
{
    for (unsigned phyQueId = 0; phyQueId < rStats.size(); phyQueId++)
    {
        if ((reportAmount > 0) && (rStats[phyQueId].mAmount >= (rStats[phyQueId].mAmountReported + reportAmount)))
        {
            dumpStatistics(phyQueId, rStats[phyQueId]);
            rStats[phyQueId].mAmountReported = rStats[phyQueId].mAmount;
        }
    }
}

void WorkCompletionManager::storeStat(const PhysicalQueuesIdListenerCsQueueArr& rCsHandles, StatArray& rStats)
{
    for (unsigned phyQueId = 0; phyQueId < rCsHandles.size(); phyQueId++)
    {
        const ListenerCsQueue& rPhyListenerCsQueue = rCsHandles[phyQueId];
        rStats[phyQueId].mAmount += rPhyListenerCsQueue.size();
        DurationNs durationTotal = 0;
        DurationNs durationMax   = 0;
        for (unsigned iter = 0; iter < rPhyListenerCsQueue.size(); iter++)
        {
            const Time& timeAdd  = std::get<2>(rPhyListenerCsQueue[iter]);
            DurationNs  duration = TimeTools::timeFromNs(timeAdd);
            durationTotal += duration;
            if (durationMax < duration)
            {
                durationMax = duration;
            }
        }
        rStats[phyQueId].mDurationTotal += durationTotal;
        DurationNs& rDurationMax = rStats[phyQueId].mDurationMax;
        if (rDurationMax < durationMax)
        {
            rDurationMax = durationMax;
        }
    }
}

void WorkCompletionManager::refreshTime(PhysicalQueuesIdListenerCsQueueArr& rCsHandles)
{
    const TimeTools::StdTime timeCurrent = TimeTools::timeNow();

    for (unsigned phyQueId = 0; phyQueId < rCsHandles.size(); phyQueId++)
    {
        ListenerCsQueue& rPhyListenerCsQueue = rCsHandles[phyQueId];
        for (unsigned iter = 0; iter < rPhyListenerCsQueue.size(); iter++)
        {
            std::get<2>(rPhyListenerCsQueue[iter]) = timeCurrent;
        }
    }
}

void WorkCompletionManager::initializeCsHandles(PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles,
                                                PhysicalQueuesIdArray&              rPhyQueues,
                                                ObserverArray&                      rObservers,
                                                CsHandleArray&                      rCsHandles,
                                                TimeArray&                          rTimes,
                                                uint32_t&                           rAmount)
{
    while (true)
    {
        bool allQueuesAreEmpty = true;
        for (unsigned phyQueId = 0; phyQueId < rIncompleteCsHandles.size(); phyQueId++)
        {
            ListenerCsQueue& rPhyListenerCsQueue = rIncompleteCsHandles[phyQueId];
            const unsigned   queueSize           = rPhyListenerCsQueue.size();
            if (queueSize == 0)
            {
                continue;
            }
            ListenerCs& rPhyListenerCs = rPhyListenerCsQueue.front();
            rPhyQueues[rAmount]        = phyQueId;
            rObservers[rAmount]        = std::get<0>(rPhyListenerCs);
            rCsHandles[rAmount]        = std::get<1>(rPhyListenerCs);
            rTimes[rAmount]            = std::get<2>(rPhyListenerCs);
            rPhyListenerCsQueue.pop_front();
            rAmount++;

            if (rAmount == WCM_HANDLES_AMOUNT_FOR_MULTI_CS)
            {
                return;
            }

            if (queueSize > 1)
            {
                allQueuesAreEmpty = false;
            }
        }
        if (allQueuesAreEmpty)
        {
            break;
        }
    }
}

void WorkCompletionManager::recycleCsHandles(PhysicalQueuesIdListenerCsQueueArr& rQueryCsHandles,
                                             bool                                csFailed,
                                             PhysicalQueuesIdListenerCsQueueArr& rCsHandles)
{
    for (unsigned phyQueId = 0; phyQueId < rQueryCsHandles.size(); phyQueId++)
    {
        ListenerCsQueueMap listenerCsQueueMap;
        ListenerCsQueue&   rPhyListenerCsQueue = rQueryCsHandles[phyQueId];

        for (ListenerCsQueue::const_reference rPhyListenerCs : rPhyListenerCsQueue)
        {
            WcmObserverInterface* listener = std::get<0>(rPhyListenerCs);
            WcmCsHandle           csHandle = std::get<1>(rPhyListenerCs);

            ListenerCsQueueMap::iterator iter = listenerCsQueueMap.find(listener);
            if (iter != listenerCsQueueMap.end())
            {
                iter->second.push_back(csHandle);
            }
            else
            {
                WcmCsHandleQueue csQueue;
                csQueue.push_back(csHandle);
                listenerCsQueueMap.insert({listener, csQueue});
            }
        }

        for (ListenerCsQueueMap::const_reference listenerCsQueue : listenerCsQueueMap)
        {
            WcmObserverInterface*   pListener  = listenerCsQueue.first;
            const WcmCsHandleQueue& rCsHandles = listenerCsQueue.second;

            for (WcmCsHandleQueue::const_reference csHandle : rCsHandles)
            {
                LOG_TRACE(SYN_WORK_COMPL,
                          "{} phyQueId {} pListener 0x{:x} csHandle {}",
                          HLLOG_FUNC,
                          phyQueId,
                          TO64(pListener),
                          csHandle);
            }

            pListener->notifyCsCompleted(rCsHandles, csFailed);
            LOG_TRACE(SYN_WORK_COMPL,
                      "{} phyQueId {} pListener 0x{:x} csHandlesSize {}",
                      HLLOG_FUNC,
                      phyQueId,
                      TO64(pListener),
                      rCsHandles.size());
        }

        storeCsHandlesQueue(rCsHandles, phyQueId, rPhyListenerCsQueue, false);
    }
}

void WorkCompletionManager::storePhysicalArr(PhysicalQueuesIdListenerCsQueueArr& rSrc,
                                             PhysicalQueuesIdListenerCsQueueArr& rDst,
                                             bool                                isFront)
{
    for (unsigned phyQueId = 0; phyQueId < rSrc.size(); phyQueId++)
    {
        ListenerCsQueue& rPhyListenerCsQueue = rSrc[phyQueId];
        storeCsHandlesQueue(rDst, phyQueId, rPhyListenerCsQueue, isFront);
    }
}

void WorkCompletionManager::dumpCsHandles(bool isWarn, const PhysicalQueuesIdListenerCsQueueArr& rCsHandles)
{
    for (unsigned phyQueId = 0; phyQueId < rCsHandles.size(); phyQueId++)
    {
        for (ListenerCsQueue::const_reference csHandle : rCsHandles[phyQueId])
        {
            const DurationNs duration = TimeTools::timeFromNs(std::get<2>(csHandle));
            if (isWarn)
            {
                LOG_WARN(SYN_WORK_COMPL,
                         "phyQueId {} pListener 0x{:x} csHandle {} Duration (ns) {}",
                         phyQueId,
                         TO64(std::get<0>(csHandle)),
                         std::get<1>(csHandle),
                         duration);
            }
            else
            {
                LOG_DEBUG(SYN_WORK_COMPL,
                          "phyQueId {} pListener 0x{:x} csHandle {} Duration (ns) {}",
                          phyQueId,
                          TO64(std::get<0>(csHandle)),
                          std::get<1>(csHandle),
                          duration);
            }
        }
    }
}

void WorkCompletionManager::dumpCsHandlesName(bool isWarn, const std::string& rCsHandlesName)
{
    if (isWarn)
    {
        LOG_WARN(SYN_WORK_COMPL, "{}", rCsHandlesName);
    }
    else
    {
        LOG_DEBUG(SYN_WORK_COMPL, "{}", rCsHandlesName);
    }
}

void WorkCompletionManager::dumpCsHandles(bool                                      isWarn,
                                          const PhysicalQueuesIdListenerCsQueueArr& rQueryIncompleteCsHandles,
                                          const PhysicalQueuesIdListenerCsQueueArr& rQueryCompleteCsHandles,
                                          const PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles,
                                          const PhysicalQueuesIdListenerCsQueueArr& rCompleteCsHandles)

{
    dumpCsHandlesName(isWarn, "rQueryIncompleteCsHandles");
    dumpCsHandles(isWarn, rQueryIncompleteCsHandles);
    dumpCsHandlesName(isWarn, "rQueryCompleteCsHandles");
    dumpCsHandles(isWarn, rQueryCompleteCsHandles);
    dumpCsHandlesName(isWarn, "rIncompleteCsHandles");
    dumpCsHandles(isWarn, rIncompleteCsHandles);
    dumpCsHandlesName(isWarn, "rCompleteCsHandles");
    dumpCsHandles(isWarn, rCompleteCsHandles);
}
