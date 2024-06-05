#pragma once

#include "queue_dispatcher.h"


namespace gaudi
{

class MmeDispatcher : public QueueDispatcher
{
public:
    MmeDispatcher(uint16_t sendSyncEventsMask, HabanaGraph* g);

    virtual void dispatchNode(const pNode& n, HabanaGraph* g, bool isSetup) override;

protected:
    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) override;

private:
    MmeDispatcher(const MmeDispatcher&) = delete;
    MmeDispatcher& operator=(const MmeDispatcher&) = delete;
};

} // namespace gaudi
