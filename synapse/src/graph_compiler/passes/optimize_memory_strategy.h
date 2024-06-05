#pragma once

#include "habana_graph.h"

class MemoryStrategyManager
{
public:
    MemoryStrategyManager()                           { m_remainingSramSize = 0; };
    MemoryStrategyManager(unsigned remainingSramSize) { m_remainingSramSize = remainingSramSize; };
    ~MemoryStrategyManager()                          {};
    virtual bool runOptimizeMemoryStrategy(HabanaGraph &g);

protected:
    virtual bool trial(HabanaGraph &g, MemoryStrategyParams &memParams, GraphAnnotation &outAnnotations) = 0;
    virtual unsigned computeIOsBufferSize(HabanaGraph &g);

private:
    unsigned m_remainingSramSize;

    unsigned computeIosBufferSizeOfNodes(HabanaGraph& g, const NodeVector& nodes);
    unsigned computeIosBufferSizeOfNodes(HabanaGraph &g, const NodeSet &nodes);
    void computeCapacityAndIOSize(HabanaGraph &g, MemoryStrategyParams &memParams);
    bool setMemoryRegions(HabanaGraph &g, MemoryStrategyParams &memParams);
    bool allocAllPersistentInSramTrial(HabanaGraph &g, MemoryStrategyParams &memParams, GraphAnnotation &outAnnotations);
    bool allocPrefetchStaticTensorsTrial(HabanaGraph &g, MemoryStrategyParams &memParams, GraphAnnotation &outAnnotations);
    bool checkAndConfigurePresetMemoryStrategy(HabanaGraph &g);
    bool runOptimizationTrials(HabanaGraph &g, MemoryStrategyParams &memParams, GraphAnnotation &ann);
};
