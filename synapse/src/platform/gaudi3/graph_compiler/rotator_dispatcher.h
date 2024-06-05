#pragma once

#include "queue_dispatcher.h"
#include "descriptor_wrapper.h"
#include "gaudi3_types.h"

namespace gaudi3
{
class RotatorDispatcher : public QueueDispatcher
{
public:
    RotatorDispatcher(uint16_t sendSyncEventsMask, HabanaGraph* g);
    RotatorDispatcher(const RotatorDispatcher&) = delete;
    RotatorDispatcher& operator=(const RotatorDispatcher&) = delete;

    virtual void dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup) override;

protected:
    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) override;

    void updateEmptyJobDescWrapper(void* wrapper) override;

private:
    RotatorDesc  m_emptyJobDesc;
    ValidityMask<RotatorDesc> m_descMask;
    Settable<deviceAddrOffset> m_inDramAddr;
    Settable<deviceAddrOffset> m_outDramAddr;
};

}  // namespace gaudi3
