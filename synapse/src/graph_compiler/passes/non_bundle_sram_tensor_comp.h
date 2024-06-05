#pragma once

#include "types.h"

class HabanaGraph;

class NonBundleSramTensorComp
{
public:
    NonBundleSramTensorComp(const HabanaGraph* graph);

    bool operator ()(const pTensor &tensor);

    static uint64_t getGraphNonBundleTensorSramSize(const HabanaGraph* graph);

private:
    static uint64_t    getMaxLiveSramCapacityForNonBundleTensors(const HabanaGraph* graph);
    const HabanaGraph* m_graph;
};
