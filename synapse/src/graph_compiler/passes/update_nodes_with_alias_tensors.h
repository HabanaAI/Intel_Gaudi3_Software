#pragma once

#include "types.h"

class HabanaGraph;


bool updateNodesWithAliasTensors(HabanaGraph& graph);

struct AliasTensors
{
    static void updateNodesWithAliasTensors(const TPCNodePtr& tpcNode);
};
