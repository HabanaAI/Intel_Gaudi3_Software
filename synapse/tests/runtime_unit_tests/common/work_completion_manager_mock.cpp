#include "work_completion_manager_mock.hpp"
#include <gtest/gtest.h>
#include "runtime/qman/common/wcm/wcm_observer_interface.hpp"
#include <algorithm>

WorkCompletionManagerMock::WorkCompletionManagerMock() : mAddCsCounter(0) {}

void WorkCompletionManagerMock::addCs(WcmPhysicalQueuesId   phyQueId,
                                      WcmObserverInterface* pObserver,
                                      WcmCsHandle           csHandle)
{
    m_pObserver = pObserver;
    m_csHandles.push_back(csHandle);
    mAddCsCounter++;
}

void WorkCompletionManagerMock::dump() {}

void WorkCompletionManagerMock::notifyCsCompleted(WcmCsHandle csHandle, bool csFailed)
{
    WcmCsHandleQueue::const_iterator iter = std::find(m_csHandles.begin(), m_csHandles.end(), csHandle);
    EXPECT_NE(iter, m_csHandles.end());

    m_pObserver->notifyCsCompleted({csHandle}, csFailed);
    m_csHandles.erase(iter);
}

void WorkCompletionManagerMock::notifyCsCompleted(bool csFailed)
{
    if (!m_csHandles.empty())
    {
        m_pObserver->notifyCsCompleted(m_csHandles, csFailed);
        m_csHandles.clear();
    }
}