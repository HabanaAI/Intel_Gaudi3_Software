#pragma once

#include "work_completion_manager_interface.hpp"
#include "statistics.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include "utils.h"
#include "timer.h"

class WcmCsQuerierInterface;

class WorkCompletionManager : public WorkCompletionManagerInterface
{
private:
    enum class StatPoints
    {
        wcmNoWork,
        wcmQuery,
        wcmQueryCsAmount,
        wcmQueryFail,
        wcmQueryTimeout,
        wcmQueryComplete,
        LAST
    };

    static constexpr auto enumNamePoints =
        toStatArray<StatPoints>({{StatPoints::wcmNoWork, "wcmNoWork"},
                                 {StatPoints::wcmQuery, "wcmQuery"},
                                 {StatPoints::wcmQueryCsAmount, "wcmQueryCsAmount"},
                                 {StatPoints::wcmQueryFail, "wcmQueryFail"},
                                 {StatPoints::wcmQueryTimeout, "wcmQueryTimeout"},
                                 {StatPoints::wcmQueryComplete, "wcmQueryComplete"}});

public:
    WorkCompletionManager(WcmCsQuerierInterface* waiter);

    virtual ~WorkCompletionManager();

    void start();

    void stop();

    virtual void addCs(WcmPhysicalQueuesId phyQueId, WcmObserverInterface* pObserver, WcmCsHandle csHandle) override;

    virtual void dump() override;

private:
    enum : uint64_t
    {
        WCM_HANDLES_AMOUNT_FOR_MULTI_CS = 32
    };
    using PhysicalQueuesIdArray              = std::array<WcmPhysicalQueuesId, WCM_HANDLES_AMOUNT_FOR_MULTI_CS>;
    using ObserverArray                      = std::array<WcmObserverInterface*, WCM_HANDLES_AMOUNT_FOR_MULTI_CS>;
    using CsHandleArray                      = std::array<WcmCsHandle, WCM_HANDLES_AMOUNT_FOR_MULTI_CS>;
    using Time                               = TimeTools::StdTime;
    using TimeArray                          = std::array<Time, WCM_HANDLES_AMOUNT_FOR_MULTI_CS>;
    using ListenerCs                         = std::tuple<WcmObserverInterface*, WcmCsHandle, Time>;
    using ListenerCsQueue                    = std::deque<ListenerCs>;
    using ListenerCsQueueMap                 = std::map<WcmObserverInterface*, WcmCsHandleQueue>;
    using PhysicalQueuesIdListenerCsQueueArr = std::array<ListenerCsQueue, QUEUE_ID_SIZE_MAX>;
    using CsCounter                          = uint64_t;
    using DurationNs                         = uint64_t;
    struct CsStat
    {
        CsCounter  mAmount;
        DurationNs mDurationTotal;
        DurationNs mDurationMax;
    };
    struct CsStatReported : public CsStat
    {
        CsCounter mAmountReported;
    };

    using StatArray = std::array<CsStatReported, QUEUE_ID_SIZE_MAX>;

    void mainLoop();

    void
    waitForEvent(bool& rWork, PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles, bool& rDump, bool& rFinish);

    void getWork(bool& rWork, PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles);

    synStatus processCsHandles(PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles);

    synStatus queryCsHandles(PhysicalQueuesIdArray&              rPhyQueues,
                             ObserverArray&                      rObservers,
                             CsHandleArray&                      rCsHandles,
                             TimeArray&                          rTimes,
                             uint32_t                            amountQuery,
                             uint64_t                            timeoutUs,
                             PhysicalQueuesIdListenerCsQueueArr& rCsIncompleteQuery,
                             PhysicalQueuesIdListenerCsQueueArr& rCsCompleteQuery,
                             uint32_t&                           rAmountCompleted);

    void processDumpStatistics() const;

    inline void dumpPreProcessedStatistics() const { dumpStatistics(m_preProcessedIncompleteCsHandlesStats); }

    inline void dumpPostProcessedStatistics() const { dumpStatistics(m_postProcessedCompleteCsHandlesStats); }

    static void dumpStatistics(const StatArray& rStats);

    static void dumpStatistics(WcmPhysicalQueuesId phyQueId, const CsStat& rStat);

    void queryCsHandlesReport();

    inline void reportPreProcessedStatistics()
    {
        reportStatistics(m_reportAmount, m_preProcessedIncompleteCsHandlesStats);
    }

    inline void reportPostProcessedStatistics()
    {
        reportStatistics(m_reportAmount, m_postProcessedCompleteCsHandlesStats);
    }

    static void reportStatistics(uint64_t reportAmount, StatArray& rStats);

    void storePreProcessedIncompleteCsHandles(PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles)
    {
        storeStat(rIncompleteCsHandles, m_preProcessedIncompleteCsHandlesStats);
        refreshTime(rIncompleteCsHandles);
    }

    void storePostProcessedCompleteCsHandles(const PhysicalQueuesIdListenerCsQueueArr& rCompleteCsHandles)
    {
        storeStat(rCompleteCsHandles, m_postProcessedCompleteCsHandlesStats);
    }

    static void storeStat(const PhysicalQueuesIdListenerCsQueueArr& rCompleteCsHandles, StatArray& rStats);

    void refreshTime(PhysicalQueuesIdListenerCsQueueArr& rCsHandles);

    static void storeCsHandle(PhysicalQueuesIdListenerCsQueueArr& rCsHandles,
                              WcmPhysicalQueuesId                 phyQueId,
                              const ListenerCs&                   rListenerCs);

    static void storeCsHandlesQueue(PhysicalQueuesIdListenerCsQueueArr& rCsHandles,
                                    WcmPhysicalQueuesId                 phyQueId,
                                    ListenerCsQueue&                    rListenerCsQueue,
                                    bool                                isFront);

    static void storeCsHandlesArr(PhysicalQueuesIdListenerCsQueueArr& rQueryIncompleteCsHandles,
                                  const PhysicalQueuesIdArray&        rPhyQueues,
                                  const ObserverArray&                rObservers,
                                  const CsHandleArray&                rCsHandles,
                                  const TimeArray&                    rTimes,
                                  uint32_t                            amount);

    void initializeCsHandles(PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles,
                             PhysicalQueuesIdArray&              rPhyQueues,
                             ObserverArray&                      rObservers,
                             CsHandleArray&                      rCsHandles,
                             TimeArray&                          rTimes,
                             uint32_t&                           rAmount);

    void recycleCsHandles(PhysicalQueuesIdListenerCsQueueArr& rQueryCompleteCsHandles,
                          bool                                csFailed,
                          PhysicalQueuesIdListenerCsQueueArr& rCompleteCsHandles);

    inline void handleCompleteCsHandles(PhysicalQueuesIdListenerCsQueueArr& rQueryCompleteCsHandles,
                                        PhysicalQueuesIdListenerCsQueueArr& rCompleteCsHandles)
    {
        recycleCsHandles(rQueryCompleteCsHandles, false, rCompleteCsHandles);
    };

    inline void handleInflightCsHandles(PhysicalQueuesIdListenerCsQueueArr& rQueryIncompleteCsHandles,
                                        PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles)
    {
        storePhysicalArr(rQueryIncompleteCsHandles, rIncompleteCsHandles, true);
    };

    inline void handleRejectedCsHandles(PhysicalQueuesIdListenerCsQueueArr& rQueryIncompleteCsHandles,
                                        PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles)
    {
        recycleCsHandles(rQueryIncompleteCsHandles, true, rIncompleteCsHandles);
    };

    static void
    storePhysicalArr(PhysicalQueuesIdListenerCsQueueArr& rSrc, PhysicalQueuesIdListenerCsQueueArr& rDst, bool isFront);

    static void dumpCsHandles(bool isWarn, const PhysicalQueuesIdListenerCsQueueArr& rCsHandles);

    static void dumpCsHandlesName(bool isWarn, const std::string& rCsHandlesName);

    static void dumpCsHandles(bool                                      isWarn,
                              const PhysicalQueuesIdListenerCsQueueArr& rQueryIncompleteCsHandles,
                              const PhysicalQueuesIdListenerCsQueueArr& rQueryCompleteCsHandles,
                              const PhysicalQueuesIdListenerCsQueueArr& rIncompleteCsHandles,
                              const PhysicalQueuesIdListenerCsQueueArr& rCompleteCsHandles);

    WcmCsQuerierInterface*     m_pQuerier;
    Statistics<enumNamePoints> m_stat;
    const uint64_t             m_reportAmount;
    std::thread*               m_thread;

    enum : uint64_t
    {
        OPERATION_WORK   = 0x01,
        OPERATION_DUMP   = 0x02,
        OPERATION_FINISH = 0x04
    };

    uint64_t m_operation;

    std::mutex              m_mutex;
    std::condition_variable m_cv;

    PhysicalQueuesIdListenerCsQueueArr m_incompleteCsHandles;

    StatArray m_preProcessedIncompleteCsHandlesStats;
    StatArray m_postProcessedCompleteCsHandlesStats;
};
