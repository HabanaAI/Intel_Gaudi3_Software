#pragma once 

#include "queue_dispatcher.h"
#include "descriptor_wrapper.h"


namespace gaudi
{

class TPCDispatcher : public QueueDispatcher
{
public:
    TPCDispatcher(unsigned activatedTpcEnginesMask, uint16_t sendSyncEventsMask, HabanaGraph* g);
    virtual ~TPCDispatcher();

    virtual void dispatchNode(const pNode& n, HabanaGraph* g, bool isSetup) override;

protected:
    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) override;

private:
    TPCDispatcher(const TPCDispatcher&) = delete;
    TPCDispatcher& operator=(const TPCDispatcher&) = delete;
};

} // namespace gaudi
