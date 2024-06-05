#pragma once

#include "queue_dispatcher.h"
#include "descriptor_wrapper.h"
#include "gaudi3_types.h"

namespace gaudi3
{
class TpcDispatcher : public QueueDispatcher
{
public:
    TpcDispatcher(unsigned activatedTpcEnginesMask, uint16_t sendSyncEventsMask, HabanaGraph* g);
    TpcDispatcher(const TpcDispatcher&) = delete;
    TpcDispatcher& operator=(const TpcDispatcher&) = delete;

    virtual void dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup) override;

protected:
    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) override;
    virtual const TpcDescriptorsWrappers& getWrappers(HabanaGraph& g, const NodePtr& n) const;
};

}  // namespace gaudi3
