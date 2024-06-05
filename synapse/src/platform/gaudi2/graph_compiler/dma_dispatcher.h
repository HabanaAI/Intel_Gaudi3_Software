#pragma once

#include "queue_dispatcher.h"
#include "gaudi2_types.h"

namespace gaudi2
{

class DmaDispatcher : public QueueDispatcher
{
public:
    DmaDispatcher(uint8_t activatedDmaEnginesMask,
                  uint16_t sendSyncEventsMask,
                  unsigned dispatcherIndex,
                  HabanaGraph* g);
    DmaDispatcher(const DmaDispatcher&) = delete;
    DmaDispatcher& operator=(const DmaDispatcher&) = delete;

    virtual void dispatchNode(const NodePtr& node, HabanaGraph* g, bool isSetup) override;

protected:
    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) override;
    virtual const DmaDescriptorsWrappers& getWrappers(HabanaGraph& g, const NodePtr& n) const;

private:
    unsigned m_numEngines;
    unsigned m_dispatcherIndex;
    DmaDesc  m_emptyJobDesc;
};

} // namespace gaudi2
