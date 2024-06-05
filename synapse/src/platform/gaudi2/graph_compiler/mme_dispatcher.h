#pragma once

#include "queue_dispatcher.h"
#include "gaudi2_types.h"

namespace gaudi2
{

class MmeDispatcher : public QueueDispatcher
{
public:
    MmeDispatcher(uint16_t sendSyncEventsMask, HabanaGraph* g);
    MmeDispatcher(const MmeDispatcher&) = delete;
    MmeDispatcher& operator=(const MmeDispatcher&) = delete;

    virtual void dispatchNode(const NodePtr& n, HabanaGraph* g, bool isSetup) override;

protected:
    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) override;
    virtual const MmeDescriptorsWrappers& getWrappers(HabanaGraph& g, const NodePtr& n) const;
};

} // namespace gaudi2
