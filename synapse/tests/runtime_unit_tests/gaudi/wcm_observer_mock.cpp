#include "wcm_observer_mock.hpp"
#include "defs.h"

WcmObserverRecorderMock::WcmObserverRecorderMock(unsigned notifyNum) : m_notifyNum(notifyNum), m_notifyNumMax(notifyNum)
{
}

void WcmObserverRecorderMock::notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed)
{
    HB_ASSERT(rCsHandles.size() > 0, "{}: Failure to few notification", __FUNCTION__);
    HB_ASSERT(m_notifyNum >= rCsHandles.size(), "{}: Failure to many notification", __FUNCTION__);
    m_notifyNum -= rCsHandles.size();

    if (m_notifyNum == 0)
    {
        m_completedCsdcAmount.set_value(m_notifyNumMax);
    }
}

void WcmObserverRecorderMock::addCs(WcmCsHandleQueue& rCsHandles) {}

WcmObserverCheckerMock::WcmObserverCheckerMock(const std::set<uint64_t>& rCompletedCsHandles)
: m_completedCsHandles(rCompletedCsHandles)
{
}

WcmObserverCheckerMock::~WcmObserverCheckerMock()
{
    if (!m_completedCsHandles.empty())
    {
        LOG_CRITICAL(GC, "{}: m_completedCsHandles {} is not empty", HLLOG_FUNC, m_completedCsHandles.size());
        std::terminate();
    }
}

void WcmObserverCheckerMock::notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed)
{
    WcmCsHandleQueue csHandles {rCsHandles};
    while (!csHandles.empty())
    {
        m_completedCsHandles.erase(m_completedCsHandles.find(csHandles.front()));
        csHandles.pop_front();
    }
}

WcmObserverAdvanceCheckerMock::WcmObserverAdvanceCheckerMock(const std::deque<std::set<uint64_t>>& rCompletedCsHandles)
: m_completedCsHandles(rCompletedCsHandles), m_csFailed(false)
{
}

WcmObserverAdvanceCheckerMock::WcmObserverAdvanceCheckerMock(const std::deque<std::set<uint64_t>>& rCompletedCsHandles,
                                                             bool                                  csFailed)
: m_completedCsHandles(rCompletedCsHandles), m_csFailed(csFailed)
{
}

WcmObserverAdvanceCheckerMock::~WcmObserverAdvanceCheckerMock()
{
    if (!m_completedCsHandles.empty())
    {
        LOG_CRITICAL(GC, "{}: m_completedCsHandles {} is not empty", HLLOG_FUNC, m_completedCsHandles.size());
        std::terminate();
    }
}

void WcmObserverAdvanceCheckerMock::notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed)
{
    HB_ASSERT(!m_completedCsHandles.empty(), "{}: Failure called with an empty m_csHandlesExpected", __FUNCTION__);
    HB_ASSERT(m_csFailed == csFailed,
              "{}: csFailed mismatch: expected {} actual {}",
              __FUNCTION__,
              m_csFailed,
              csFailed);

    const size_t expected = m_completedCsHandles.front().size();
    const size_t actual   = rCsHandles.size();
    HB_ASSERT(expected == actual, "{}: size mismatch expected {} actual {}", __FUNCTION__, expected, actual);

    for (WcmCsHandleQueue::const_reference actualCs : rCsHandles)
    {
        HB_ASSERT(m_completedCsHandles.front().count(actualCs) == 1,
                  "{}: actualCs {} does not exist",
                  __FUNCTION__,
                  actualCs);
    }

    m_completedCsHandles.pop_front();
}
