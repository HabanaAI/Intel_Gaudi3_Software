#pragma once

#include "runtime/qman/common/wcm/work_completion_manager_interface.hpp"

class WorkCompletionManagerMock : public WorkCompletionManagerInterface
{
public:
    WorkCompletionManagerMock();

    virtual ~WorkCompletionManagerMock() override = default;

    virtual void addCs(WcmPhysicalQueuesId phyQueId, WcmObserverInterface* pObserver, WcmCsHandle csHandle) override;

    virtual void dump() override;

    void notifyCsCompleted(WcmCsHandle csHandle, bool csFailed);

    void notifyCsCompleted(bool csFailed);

    uint64_t mAddCsCounter;

    WcmObserverInterface* m_pObserver;

    WcmCsHandleQueue m_csHandles;
};
