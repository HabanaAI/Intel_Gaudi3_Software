#pragma once
#include "runtime/qman/common/wcm/wcm_observer_interface.hpp"
#include <set>
#include <deque>
#include <future>

class WcmObserverRecorderMock : public WcmObserverInterface
{
public:
    WcmObserverRecorderMock(unsigned notifyNum);

    virtual void notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed) override;

    void addCs(WcmCsHandleQueue& rCsHandles);

    std::promise<unsigned> m_completedCsdcAmount;
    unsigned               m_notifyNum;
    const unsigned         m_notifyNumMax;
};

class WcmObserverCheckerMock : public WcmObserverInterface
{
public:
    WcmObserverCheckerMock(const std::set<uint64_t>& rCompletedCsHandles);
    virtual ~WcmObserverCheckerMock();

    virtual void notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed) override;

    std::set<uint64_t> m_completedCsHandles;
};

class WcmObserverAdvanceCheckerMock : public WcmObserverInterface
{
public:
    WcmObserverAdvanceCheckerMock(const std::deque<std::set<uint64_t>>& rCompletedCsHandles);
    WcmObserverAdvanceCheckerMock(const std::deque<std::set<uint64_t>>& rCompletedCsHandles, bool csFailed);
    virtual ~WcmObserverAdvanceCheckerMock();

    virtual void notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed) override;

    std::deque<std::set<uint64_t>> m_completedCsHandles;
    bool                           m_csFailed;
};
