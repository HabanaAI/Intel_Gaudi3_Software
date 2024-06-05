#pragma once

#include <set>
#include "queue_dispatcher.h"

class HabanaGraph;

class DmaDispatcherCreator
{
public:
    DmaDispatcherCreator() = default;
    virtual bool go(HabanaGraph& g);

protected:
    uint8_t                    getAvailableEnginesMask(HabanaGraph& g, unsigned parallelLevel) const;
    virtual QueueDispatcherPtr makeDmaDispatcher(uint8_t mask, unsigned dispatcherIndex, HabanaGraph* g) const = 0;
    virtual bool isParallelLevelsValid(std::set<QueueDispatcherParams> allParallelLevels) const { return true; }
};
