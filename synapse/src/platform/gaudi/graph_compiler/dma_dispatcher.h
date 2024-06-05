#pragma once

#include "queue_dispatcher.h"

namespace gaudi
{

class DMADispatcher : public QueueDispatcher
{
public:
    DMADispatcher(uint8_t activatedDmaEnginesMask,
                  uint16_t sendSyncEventsMask,
                  unsigned dispatcherIndex,
                  HabanaGraph* g);
    virtual ~DMADispatcher();

    virtual void dispatchNode(const pNode& node, HabanaGraph* g, bool isSetup) override;

    virtual void
    addEmptyJob(const NodePtr& n, uint32_t pipeLevel, CommandQueuePtr queue, bool isLastPipelineLevel) override;

protected:
    virtual CommandQueue* createCommandQueue(uint32_t queueId, uint32_t engineIdx, HabanaGraph* g) override;

private:
    DMADispatcher(const DMADispatcher&) = delete;
    DMADispatcher& operator=(const DMADispatcher&) = delete;

    unsigned m_numEngines;
    unsigned m_dispatcherIndex;
};

} // namespace gaudi
