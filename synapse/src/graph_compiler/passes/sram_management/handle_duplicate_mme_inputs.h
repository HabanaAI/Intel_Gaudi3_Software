#pragma once

#include "habana_graph.h"

class DuplicateMmeInputsHandler
{
public:
    static void handleDuplicateMmeInputs(HabanaGraph& g);
};