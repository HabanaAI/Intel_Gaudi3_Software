#include "habana_graph.h"

bool checkMaxDims(HabanaGraph& graph)
{
    auto deviceType = graph.getDeviceType();
    for (const NodePtr& node : graph.getTopoSortedNodes())
    {
        if (node->isDynamicShape())
        {
            LOG_TRACE(GC, "Max dims checking for node \"{}\"", node->getNodeName());
            if (!node->inferOutputsSizes(deviceType, /*inferMax*/ true))
            {
                LOG_ERR(GC, "Failure to update output shape for node: \"{}\"", node->getNodeName());
                return false;
            }
        }
    }
    return true;
}

bool checkMaxDimsPreCompilation(HabanaGraph& graph)
{
    return checkMaxDims(graph);
}

bool checkMaxDimsPostCompilation(HabanaGraph& graph)
{
    return checkMaxDims(graph);
}

bool nodeCreatedWithoutOutputShape(HabanaGraph& graph)
{
    return checkMaxDims(graph);
}