#pragma once

#include "../../../../graph_compiler/dma_dispatchers_creator.h"

namespace gaudi2
{

class DmaDispatcherCreator : public ::DmaDispatcherCreator
{
public:
    DmaDispatcherCreator() = default;
protected:
    virtual QueueDispatcherPtr makeDmaDispatcher(uint8_t mask, unsigned dispatcherIndex, HabanaGraph* g) const override;
    virtual bool               isParallelLevelsValid(std::set<QueueDispatcherParams> allParallelLevels) const override;
};

} // namespace gaudi2
